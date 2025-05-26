import torch
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from typing import Dict, Union, Any, Optional, Tuple, List
import os 
import wandb # Make sure wandb is imported

logger = logging.getLogger(__name__)

class Rho1Trainer(Trainer):
    def __init__(self, *args, reference_model_name_or_path=None, beta_rho=0.1, slm_top_k_ratio=None, **kwargs): # Changed slm_threshold to slm_top_k_ratio
        super().__init__(*args, **kwargs)
        self.beta_rho = beta_rho
        self.slm_top_k_ratio = slm_top_k_ratio # Store the top_k_ratio

        self.reference_model = None

        if reference_model_name_or_path:
            logger.info(f"Loading reference model from {reference_model_name_or_path}")
            self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name_or_path)
            self.reference_model.to(self.args.device)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        else:
            logger.warning("Reference model not provided. Rho-1 loss and SLM (if based on ref model) cannot be computed if enabled.")
            # Check if reference model is needed
            if self.beta_rho > 0 or (self.slm_top_k_ratio is not None and self.slm_top_k_ratio > 0):
                raise ValueError("Reference model path must be provided if beta_rho > 0 or slm_top_k_ratio is set and > 0.")


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss.
        If slm_top_k_ratio is set, incorporates Selective Language Modeling (SLM) 
        by selecting top-k tokens based on excess loss.
        """
        labels = inputs.pop("labels", None)
        
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        shift_logits_student = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct_ce = torch.nn.CrossEntropyLoss(reduction='none')
        
        loss_lm_full_flat = loss_fct_ce(
            shift_logits_student.view(-1, shift_logits_student.size(-1)), 
            shift_labels.view(-1)
        )
        loss_lm_full = loss_lm_full_flat.view(shift_labels.shape[0], shift_labels.shape[1]) # Student's per-token loss

        active_loss_mask = shift_labels.ne(-100) # Mask for non-padding tokens
        num_active_tokens_total = active_loss_mask.sum() # Total number of active tokens in the batch

        if num_active_tokens_total > 0:
            base_ce_loss = (loss_lm_full * active_loss_mask).sum() / num_active_tokens_total
        else:
            base_ce_loss = torch.tensor(0.0, device=student_logits.device)
        
        current_loss = base_ce_loss
        
        # --- Selective Language Modeling (SLM) based on Top-K Excess Loss ---
        if self.reference_model and self.slm_top_k_ratio is not None and self.slm_top_k_ratio > 0:
            with torch.no_grad(): # Operations involving the reference model should not compute gradients
                ref_model_inputs = {
                    "input_ids": inputs["input_ids"].clone(), 
                    "attention_mask": inputs["attention_mask"].clone()
                }
                reference_outputs = self.reference_model(**ref_model_inputs)
                reference_logits = reference_outputs.logits.detach() # (batch_size, seq_len, vocab_size)
                
                shift_logits_ref = reference_logits[..., :-1, :].contiguous()

                # Calculate per-token Cross-Entropy loss for the reference model
                ref_loss_flat = loss_fct_ce(
                    shift_logits_ref.view(-1, shift_logits_ref.size(-1)), 
                    shift_labels.view(-1) # Use the same ground truth labels
                )
                ref_loss_token_wise = ref_loss_flat.view(shift_labels.shape[0], shift_labels.shape[1])
                ref_loss_token_wise[~active_loss_mask] = 0.0 # Mask out padding tokens for reference loss

            # Calculate excess loss: L_student_token_wise - L_reference_token_wise
            # loss_lm_full has gradients from the student model.
            # ref_loss_token_wise is detached (no gradients).
            excess_loss = loss_lm_full - ref_loss_token_wise
            
            slm_selection_mask = torch.zeros_like(excess_loss, dtype=torch.bool, device=excess_loss.device)
            
            for i in range(excess_loss.size(0)): # Iterate over each sequence in the batch
                num_active_tokens_in_sequence = active_loss_mask[i].sum().item()
                
                if num_active_tokens_in_sequence == 0:
                    continue # Skip if no active tokens in this sequence

                # Determine the number of top-k tokens to select for this sequence
                num_tokens_to_select = int(self.slm_top_k_ratio * num_active_tokens_in_sequence)
                # Ensure num_tokens_to_select is not more than available active tokens, and at least 0
                num_tokens_to_select = min(max(0, num_tokens_to_select), num_active_tokens_in_sequence)

                if num_tokens_to_select > 0:
                    # Create a copy of excess_loss for this sequence to modify for topk selection
                    current_sequence_excess_loss = excess_loss[i].clone()
                    # Mask out non-active tokens by setting their excess_loss to a very low value
                    # so they are not selected by topk.
                    current_sequence_excess_loss[~active_loss_mask[i]] = -float('inf')
                    
                    # Get topk indices from the modified excess_loss.
                    # These indices are directly applicable to the original sequence dimension.
                    _, topk_indices = torch.topk(current_sequence_excess_loss, num_tokens_to_select)
                    slm_selection_mask[i, topk_indices] = True
            
            num_selected_tokens_slm = slm_selection_mask.sum()

            # Log SLM selection ratio if this is the main process and at a logging step
            if self.is_world_process_zero() and self.state.global_step % self.args.logging_steps == 0:
                if num_active_tokens_total > 0: # Use total active tokens in batch for overall ratio
                    wandb.log({"slm_selected_ratio": num_selected_tokens_slm.float() / num_active_tokens_total.float()}, step=self.state.global_step)
                else:
                    wandb.log({"slm_selected_ratio": 0.0}, step=self.state.global_step)

            # If SLM selected any tokens, the SLM loss (student's loss on these selected tokens)
            # becomes the current_loss.
            if num_selected_tokens_slm > 0:
                loss_slm = (loss_lm_full * slm_selection_mask).sum() / num_selected_tokens_slm
                current_loss = loss_slm 
            # Else (no tokens selected by SLM, e.g., if slm_top_k_ratio is 0 or no active tokens),
            # current_loss remains base_ce_loss (or 0 if no active tokens at all).
            elif num_active_tokens_total == 0 : # If no active tokens in the batch at all
                current_loss = torch.tensor(0.0, device=student_logits.device)
            # If active tokens exist but SLM selected none (e.g. ratio is 0), current_loss is already base_ce_loss.        
        if return_outputs:
            return (current_loss, student_outputs)
        return current_loss