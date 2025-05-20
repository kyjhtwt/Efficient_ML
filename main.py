import logging
import os
import argparse
import torch
import math
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    OPTConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import wandb

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
            wandb.log({"eval_perplexity":metrics["perplexity"], "eval_loss": metrics["eval_loss"]}, step=state.global_step)
        return control


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an OPT model on the C4 dataset using Hugging Face Trainer"
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Type of learning rate scheduler"
    )
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m",
                        help="Pretrained OPT model identifier from Hugging Face Hub")
    parser.add_argument("--language", type=str, default="en",
                        help="Language split of the C4 dataset to use")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length after tokenization")
    parser.add_argument("--output_dir", type=str, default="./opt-c4-output",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (ignored if --max_steps > 0)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Total number of training steps to perform (overrides epochs if > 0)")
    parser.add_argument("--train_batch_size", type=int, default=64,
                        help="Batch size per device for training")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size per device for evaluation")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Run evaluation every X update steps")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X update steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay to apply")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X update steps")
    parser.add_argument("--fp16", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push trained model to Hugging Face Hub")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Enable torch DataParallel for multi-GPU training")
    parser.add_argument("--wandb_project", type=str, default="Efficient_ML",
                        help="Weights & Biases project name to log to")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (optional)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--target_dataset", type=str, default="c4",
                    help="target dataset to fit")
    parser.add_argument("--target_task", type=str, default="reference_train",
                    choices=["SLM", "RHO-LOSS", "LESS", "reference_train"])

    return parser.parse_args()




def main():
    args = parse_args()

    # WandB 초기화
    if args.wandb_project:
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

    # 7) Multi-GPU 설정
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # 8) 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 9) 학습 인자 설정
    report_to = ["wandb"] if args.wandb_project else ["none"]
    # report_to = ["none"]

    if(args.target_task == "reference_train"):
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=args.logging_steps,
            fp16=args.fp16,
            bf16=False,
            push_to_hub=args.push_to_hub,
            report_to=report_to,
            run_name=args.wandb_run_name,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        # 10) Trainer 생성
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_train,
            data_collator=data_collator,
            callbacks=[PerplexityCallback()],
            eval_dataset=lm_eval
        )

    elif(args.target_task == "SLM"):
        raise NotImplementedError
        # Not all tokens ...
    elif(args.target_task == "RHO-LOSS"):
        # RHO-LOSS
        raise NotImplementedError
    elif(args.target_task == "LESS"):
        # LESS
        raise NotImplementedError
    
    # 11) 학습 시작
    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 12) 모델 저장
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
