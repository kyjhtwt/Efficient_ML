import logging
import os
import argparse
import torch
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import wandb

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.best_loss = float("inf")
        self.tokenizer = tokenizer
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Only the main process logs and saves in DDP
        if not getattr(state, "is_world_process_zero", False):
            return control

        # Retrieve evaluation loss
        eval_loss = None
        if metrics is not None:
            eval_loss = metrics.get("eval_loss") or metrics.get("loss")
        if eval_loss is None:
            print(f"Warning: evaluation loss not found in metrics: {metrics}")
            return control

        perplexity = math.exp(eval_loss)
        # Console 출력
        print(f"***** Evaluation results at step {state.global_step} *****")
        print(f"eval_loss: {eval_loss:.4f} | perplexity: {perplexity:.2f}")

        # WandB 로깅 (main process)
        wandb.log({
            "eval_loss": eval_loss,
            "eval_perplexity": perplexity
        }, step=state.global_step)

        # 베스트 모델 저장
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            out_dir = os.path.join(args.output_dir, "checkpoint-best")
            if os.path.isdir(out_dir):
                for f in os.listdir(out_dir):
                    os.remove(os.path.join(out_dir, f))
            # If model is wrapped in DDP, unwrap it
            model_to_save = kwargs["model"].module if hasattr(kwargs.get("model"), "module") else kwargs.get("model")
            model_to_save.save_pretrained(out_dir)
            # Tokenizer is not wrapped
            self.tokenizer.save_pretrained(out_dir)
            print(f">>> New best eval_loss: {eval_loss:.4f}. Saved to {out_dir}")
        else:
            control.should_save = False
        return control


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an OPT model on the C4 dataset using Hugging Face Trainer with DDP"
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear","cosine","cosine_with_restarts",
                                 "polynomial","constant","constant_with_warmup"])
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./opt-c4-output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument("--wandb_project", type=str, default="Efficient_ML")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--target_dataset", type=str, default="c4")
    parser.add_argument("--target_task", type=str, default="reference_train",
                        choices=["SLM","RHO-LOSS","LESS","reference_train"])
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # DDP 초기화
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WandB 초기화 (rank 0만)
    if args.wandb_project and rank == 0:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lm_train = None
    lm_eval = None

    if(args.target_dataset=="c4"):
        examples = []
        stream_ds = load_dataset(   
            "allenai/c4",  
            args.language,  
            split="train",  
            streaming=True  
        )
        # target_tokens = 200_000_000
        target_tokens = 2_000_000
        accum_tokens = 0
        for ex in stream_ds:
            tokenized = tokenizer(ex["text"], add_special_tokens=False)
            length = len(tokenized["input_ids"])
            if accum_tokens + length > target_tokens:
                break
            examples.append(ex)
            accum_tokens += length
        logger.info(f"Collected {len(examples)} examples ≈ {accum_tokens} tokens.")
        raw_full = Dataset.from_list(examples)
        split = raw_full.train_test_split(test_size=0.1, seed=42)
        raw_train = split["train"]
        raw_eval  = split["test"]

        def tokenize_function(examples_batch):
            return tokenizer(examples_batch['text'], return_special_tokens_mask=True)
        tokenized_train = raw_train.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_train.column_names,
            desc="Tokenizing dataset"
        )
        tokenized_eval  = raw_eval.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_eval.column_names,
            desc="Tokenizing dataset"
        )
        def group_texts(examples_batch):
            concatenated = {k: sum(examples_batch[k], []) for k in examples_batch.keys()}
            total_length = len(concatenated['input_ids'])
            total_length = (total_length // args.block_size) * args.block_size
            result = {
                k: [t[i: i + args.block_size] for i in range(0, total_length, args.block_size)]
                for k, t in concatenated.items()
            }
            result['labels'] = result['input_ids'].copy()
            return result
        lm_train = tokenized_train.map(group_texts, batched=True, desc="Grouping train")
        lm_eval  = tokenized_eval.map(group_texts, batched=True, desc="Grouping eval")

    elif(args.target_dataset =="tulu"):
        raise NotImplementedError
    elif(args.target_dataset == "GSM8K"):
        dataset = load_dataset("gsm8k", "main")
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_raw = split["train"]
        test_raw = split["test"]
        def preprocess(examples):
            inputs = []
            for q, a in zip(examples["question"], examples["answer"]):
                text = f"Question: {q}\nAnswer: {a}"
                inputs.append(text)
            tokenized = tokenizer(
                inputs,
                padding="max_length",
                truncation=True,
                max_length=args.block_size,
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        lm_train = train_raw.map(
            preprocess,
            batched=True,
            remove_columns=train_raw.column_names,
        )

        lm_eval = test_raw.map(
            preprocess,
            batched=True,
            remove_columns=test_raw.column_names,
        )

    # elif(args.target_dataset=="openhermes"):

    logger.info(f"Loading model {args.model_name}")
    model = OPTForCausalLM.from_pretrained(args.model_name)
    model.to(device)

    # DDP 래핑
    if args.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        local_rank=args.local_rank,
        report_to=["wandb"] if args.wandb_project else ["none"],
        run_name=args.wandb_run_name,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        greater_is_better=False,
        label_names=["labels"],  # ensure loss 계산
        metric_for_best_model="eval_loss"
    )
    callbacks = [SaveBestModelCallback(tokenizer=tokenizer)]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=tokenizer
    )

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 최종 저장 (rank 0만)
    if args.world_size == 1 or dist.get_rank() == 0:
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()