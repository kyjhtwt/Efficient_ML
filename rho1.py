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
    def __init__(self, *args, reference_model_name_or_path=None, beta_rho=0.0, slm_top_k_ratio=None, slm_selection_strategy="excess_loss", **kwargs): # Added slm_selection_strategy
        super().__init__(*args, **kwargs)
        self.beta_rho = beta_rho
        self.slm_top_k_ratio = slm_top_k_ratio
        self.slm_selection_strategy = slm_selection_strategy # Store the selection strategy

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
            if self.beta_rho > 0 or \
               (self.slm_top_k_ratio is not None and self.slm_top_k_ratio > 0 and self.slm_selection_strategy == "excess_loss"):
                raise ValueError("Reference model path must be provided if beta_rho > 0 or (slm_top_k_ratio > 0 and slm_selection_strategy is 'excess_loss').")


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss.
        If slm_top_k_ratio is set, incorporates Selective Language Modeling (SLM)
        by selecting top-k tokens based on the slm_selection_strategy.
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
        
        # --- Selective Language Modeling (SLM) ---
        if self.slm_top_k_ratio is not None and self.slm_top_k_ratio > 0:
            slm_selection_mask = torch.zeros_like(active_loss_mask, dtype=torch.bool, device=active_loss_mask.device)

            if self.slm_selection_strategy == "excess_loss":
                if not self.reference_model:
                    logger.error("Reference model is required for 'excess_loss' SLM strategy but not loaded.")
                    if return_outputs:
                        return (current_loss, student_outputs)
                    return current_loss
                
                # Temporary variables for reference model pass
                ref_input_ids = inputs["input_ids"].clone()
                ref_attention_mask = inputs["attention_mask"].clone()
                
                with torch.no_grad():
                    ref_model_inputs = {
                        "input_ids": ref_input_ids, 
                        "attention_mask": ref_attention_mask
                    }
                    reference_outputs = self.reference_model(**ref_model_inputs)
                    reference_logits = reference_outputs.logits.detach() # Detached, but still in memory
                    
                    # Explicitly delete cloned inputs and full outputs after use
                    del ref_input_ids, ref_attention_mask, ref_model_inputs, reference_outputs
                    # torch.cuda.empty_cache() # Optional: clear cache if memory is extremely tight

                    shift_logits_ref = reference_logits[..., :-1, :].contiguous()
                    # ref_loss_flat will be created and then reshaped to ref_loss_token_wise
                    ref_loss_flat = loss_fct_ce(
                        shift_logits_ref.view(-1, shift_logits_ref.size(-1)), 
                        shift_labels.view(-1)
                    )
                    ref_loss_token_wise = ref_loss_flat.view(shift_labels.shape[0], shift_labels.shape[1])
                    ref_loss_token_wise[~active_loss_mask] = 0.0

                    # Explicitly delete intermediate tensors for ref loss calculation
                    del reference_logits, shift_logits_ref, ref_loss_flat
                    torch.cuda.empty_cache() # Optional

                # Now calculate excess_loss using loss_lm_full (from student) and ref_loss_token_wise
                # loss_lm_full is kept as it's needed for the final SLM loss calculation if tokens are selected
                excess_loss_values = loss_lm_full - ref_loss_token_wise
                
                # ref_loss_token_wise is no longer needed after calculating excess_loss_values
                del ref_loss_token_wise
                torch.cuda.empty_cache() # Optional

                for i in range(excess_loss_values.size(0)): # Iterate over batch dimension
                    num_active_tokens_in_sequence = active_loss_mask[i].sum().item()
                    if num_active_tokens_in_sequence == 0:
                        continue
                    
                    num_tokens_to_select = min(max(0, int(self.slm_top_k_ratio * num_active_tokens_in_sequence)), num_active_tokens_in_sequence)
                    
                    if num_tokens_to_select > 0:
                        # Clone only the necessary slice for modification and topk
                        current_sequence_excess_loss_for_topk = excess_loss_values[i].clone()
                        current_sequence_excess_loss_for_topk[~active_loss_mask[i]] = -float('inf')
                        _, topk_indices = torch.topk(current_sequence_excess_loss_for_topk, num_tokens_to_select)
                        slm_selection_mask[i, topk_indices] = True
                        # Delete the cloned tensor after use in the loop
                        del current_sequence_excess_loss_for_topk
                
                # excess_loss_values is no longer needed after the loop
                del excess_loss_values
                torch.cuda.empty_cache() # Optional
            
            elif self.slm_selection_strategy == "random":
                for i in range(active_loss_mask.size(0)): # Iterate over each sequence in the batch
                    active_indices_in_sequence = torch.where(active_loss_mask[i])[0]
                    num_active_tokens_in_sequence = len(active_indices_in_sequence)

                    if num_active_tokens_in_sequence == 0:
                        continue

                    num_tokens_to_select = min(max(0, int(self.slm_top_k_ratio * num_active_tokens_in_sequence)), num_active_tokens_in_sequence)

                    if num_tokens_to_select > 0:
                        # Randomly select 'num_tokens_to_select' from the active_indices
                        perm = torch.randperm(num_active_tokens_in_sequence, device=active_indices_in_sequence.device)
                        selected_relative_indices = perm[:num_tokens_to_select]
                        selected_absolute_indices = active_indices_in_sequence[selected_relative_indices]
                        slm_selection_mask[i, selected_absolute_indices] = True
            else:
                logger.warning(f"Unknown slm_selection_strategy: {self.slm_selection_strategy}. Defaulting to full CE loss.")
                # Fallback to base_ce_loss
                if return_outputs:
                    return (current_loss, student_outputs)
                return current_loss

            num_selected_tokens_slm = slm_selection_mask.sum()

            if self.is_world_process_zero() and self.state.global_step % self.args.logging_steps == 0:
                if num_active_tokens_total > 0:
                    wandb.log({
                        f"slm_{self.slm_selection_strategy}_selected_ratio": num_selected_tokens_slm.float() / num_active_tokens_total.float()
                    }, step=self.state.global_step)
                else:
                    wandb.log({f"slm_{self.slm_selection_strategy}_selected_ratio": 0.0}, step=self.state.global_step)

            if num_selected_tokens_slm > 0:
                loss_slm = (loss_lm_full * slm_selection_mask).sum() / num_selected_tokens_slm
                current_loss = loss_slm
            elif num_active_tokens_total == 0:
                current_loss = torch.tensor(0.0, device=student_logits.device)
            # If active tokens exist but SLM selected none (e.g. ratio is 0), current_loss is already base_ce_loss.
        
        # Rho-loss (if enabled, can be combined with SLM or full CE)
        # Note: The original code did not show how beta_rho was used.
        # Assuming rho_loss would be added to current_loss if beta_rho > 0.
        # This part is kept conceptual as the original did not detail rho_loss calculation.
        if self.beta_rho > 0 and self.reference_model:
            # Placeholder for actual Rho-loss calculation.
            # For example, it might involve comparing student and reference model probabilities.
            # rho_loss_value = compute_rho_loss(student_logits, reference_logits, labels, active_loss_mask) 
            # current_loss = current_loss + self.beta_rho * rho_loss_value
            logger.debug(f"beta_rho is {self.beta_rho}, but Rho-loss calculation logic is not fully implemented in this snippet.")
            pass


        if return_outputs:
            return (current_loss, student_outputs)
        return current_loss