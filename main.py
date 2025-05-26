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
from rho1 import Rho1Trainer # Rho1Trainer 임포트

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
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--target_dataset", type=str, default="c4")
    parser.add_argument("--target_task", type=str, default="REFERENCE",
                        choices=["REFERENCE", "DENSE", "SLM"])
    ### SLM/Rho1 arguments
    parser.add_argument("--reference_model_name_or_path", type=str, default=None,
                        help="Path to the reference model for SLM or Rho-loss.")
    parser.add_argument("--slm_top_k_ratio", type=float, default=0.7,
                        help="Top-k ratio for token selection in SLM. If 0, SLM is effectively disabled unless beta_rho > 0.")
    parser.add_argument("--beta_rho", type=float, default=0.0,
                        help="Beta coefficient for Rho-loss. If 0, Rho-loss is disabled.")
    parser.add_argument("--slm_selection_strategy", type=str, default="excess_loss",
                        choices=["excess_loss", "random"],
                        help="Strategy for selecting tokens in SLM: 'excess_loss' or 'random'.")
    # AdamW optimizer parameters
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for AdamW optimizer")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_train = None
    lm_eval = None

    # Common tokenization arguments for conversational datasets
    common_tokenization_args = {
        "padding": "max_length",
        "truncation": True,
        "max_length": args.block_size,
    }

    if args.target_dataset == "c4":
        examples = []
        # Limit C4 dataset size for faster example runs if needed
        # Consider making target_tokens an argument if frequently changed
        stream_ds = load_dataset(
            "allenai/c4",
            args.language,
            split="train",
            streaming=True
        )
        target_tokens = 2_000_000 # Example: 2M tokens
        # target_tokens = 200_000_000 # Original target
        accum_tokens = 0
        logger.info(f"Streaming C4 dataset, targeting approx {target_tokens} tokens...")
        for ex in stream_ds:
            # Rough estimate of tokens without full tokenization for speed
            length = len(ex["text"].split()) 
            if accum_tokens + length > target_tokens:
                if not examples: # Ensure at least one example if target_tokens is very small
                    examples.append(ex)
                    accum_tokens += length
                break
            examples.append(ex)
            accum_tokens += length
            if len(examples) % 1000 == 0:
                logger.info(f"Collected {len(examples)} examples, approx {accum_tokens} tokens so far...")

        logger.info(f"Collected {len(examples)} examples from C4, approximating {accum_tokens} tokens.")
        if not examples:
            raise ValueError("No examples collected from C4. Check dataset streaming or target_tokens.")

        raw_full = Dataset.from_list(examples)
        split = raw_full.train_test_split(test_size=0.01, seed=42) # Smaller eval for C4 example
        raw_train = split["train"]
        raw_eval  = split["test"]

        def tokenize_function_c4(examples_batch):
            return tokenizer(examples_batch['text'], return_special_tokens_mask=True)

        tokenized_train = raw_train.map(
            tokenize_function_c4,
            batched=True,
            remove_columns=raw_train.column_names,
            desc="Tokenizing C4 train dataset"
        )
        tokenized_eval  = raw_eval.map(
            tokenize_function_c4,
            batched=True,
            remove_columns=raw_eval.column_names,
            desc="Tokenizing C4 eval dataset"
        )

        def group_texts_c4(examples_batch):
            concatenated = {k: sum(examples_batch[k], []) for k in examples_batch.keys()}
            total_length = len(concatenated['input_ids'])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // args.block_size) * args.block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + args.block_size] for i in range(0, total_length, args.block_size)]
                for k, t in concatenated.items()
            }
            result['labels'] = result['input_ids'].copy()
            return result
        lm_train = tokenized_train.map(group_texts_c4, batched=True, desc="Grouping C4 train texts")
        lm_eval  = tokenized_eval.map(group_texts_c4, batched=True, desc="Grouping C4 eval texts")

    elif args.target_dataset == "tulu":
        logger.info("Loading Tulu dataset (allenai/tulu-v2-sft-mixture)")
        dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
        # Example: Use a subset for faster runs, e.g., first 10k examples
        dataset = dataset.select(range(10000)) 
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_raw = split["train"]
        test_raw = split["test"]

        def preprocess_tulu(examples_batch):
            processed_texts = []
            for messages in examples_batch["messages"]:
                text_parts = []
                for msg in messages:
                    role = str(msg.get('role', 'unknown')).upper()
                    content = str(msg.get('content', ''))
                    text_parts.append(f"{role}: {content}")
                # Add EOS at the end of the full conversation turn
                full_text = "\n".join(text_parts) + tokenizer.eos_token
                processed_texts.append(full_text)
            
            tokenized = tokenizer(
                processed_texts,
                **common_tokenization_args
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        lm_train = train_raw.map(preprocess_tulu, batched=True, remove_columns=train_raw.column_names, desc="Processing Tulu train")
        lm_eval = test_raw.map(preprocess_tulu, batched=True, remove_columns=test_raw.column_names, desc="Processing Tulu eval")

    elif args.target_dataset == "openhermes":
        logger.info("Loading OpenHermes dataset (teknium/OpenHermes-2.5)")
        dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
        # Example: Use a subset for faster runs
        # dataset = dataset.select(range(20000)) 
        split = dataset.train_test_split(test_size=0.1, seed=42) # Adjust test_size as needed
        train_raw = split["train"]
        test_raw = split["test"]

        def preprocess_openhermes(examples_batch):
            processed_texts = []
            for conversation in examples_batch["conversations"]:
                text_parts = []
                for turn in conversation:
                    speaker = str(turn.get('from', 'unknown')).upper()
                    value = str(turn.get('value', ''))
                    text_parts.append(f"{speaker}: {value}")
                full_text = "\n".join(text_parts) + tokenizer.eos_token
                processed_texts.append(full_text)

            tokenized = tokenizer(
                processed_texts,
                **common_tokenization_args
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        lm_train = train_raw.map(preprocess_openhermes, batched=True, remove_columns=train_raw.column_names, desc="Processing OpenHermes train")
        lm_eval = test_raw.map(preprocess_openhermes, batched=True, remove_columns=test_raw.column_names, desc="Processing OpenHermes eval")


    elif args.target_dataset == "GSM8K":
        logger.info("Loading GSM8K dataset")
        dataset = load_dataset("gsm8k", "main")
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_raw = split["train"]
        test_raw = split["test"]
        def preprocess_gsm8k(examples_batch):
            inputs = []
            for q, a in zip(examples_batch["question"], examples_batch["answer"]):
                text = f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}" # Added EOS
                inputs.append(text)
            tokenized = tokenizer(
                inputs,
                **common_tokenization_args
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        lm_train = train_raw.map(
            preprocess_gsm8k,
            batched=True,
            remove_columns=train_raw.column_names,
            desc="Processing GSM8K train"
        )
        lm_eval = test_raw.map(
            preprocess_gsm8k,
            batched=True,
            remove_columns=test_raw.column_names,
            desc="Processing GSM8K eval"
        )

    elif args.target_dataset == "openwebmath":
        logger.info("Loading OpenWebMath dataset (open-web-math/open-web-math)")
        
        num_examples_to_take = 20000  # 샘플 수 지정
        try:
            # 스트리밍으로 전체 데이터셋 로드 시도
            dataset_full = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
            logger.info(f"Taking first {num_examples_to_take} examples from OpenWebMath stream.")
            # 지정된 수만큼 예제 가져오기
            examples = list(dataset_full.take(num_examples_to_take))
            if not examples:
                raise ValueError(f"No examples taken from OpenWebMath stream. Check dataset or num_examples_to_take ({num_examples_to_take}).")
            raw_dataset = Dataset.from_list(examples)
        except Exception as e:
            logger.error(f"Failed to load or stream OpenWebMath: {e}")
            logger.info("Attempting to load without streaming and select a subset (might be slow or memory intensive).")
            # 스트리밍 실패 시, 전체 로드 후 일부 선택 (메모리 주의)
            dataset_all_splits = load_dataset("open-web-math/open-web-math")
            if "train" not in dataset_all_splits:
                raise ValueError("OpenWebMath dataset does not contain 'train' split.")
            dataset_train_full = dataset_all_splits["train"]
            if len(dataset_train_full) < num_examples_to_take:
                logger.warning(f"Full dataset ({len(dataset_train_full)}) is smaller than requested num_examples_to_take ({num_examples_to_take}). Using all available examples.")
                num_examples_to_take = len(dataset_train_full)
            
            if num_examples_to_take == 0:
                 raise ValueError("No examples available in OpenWebMath dataset after attempting to load.")
            raw_dataset = dataset_train_full.select(range(num_examples_to_take))

        if len(raw_dataset) < 2:
            raise ValueError(f"OpenWebMath dataset is too small to split (found {len(raw_dataset)} examples). Adjust data loading or num_examples_to_take.")

        # 데이터셋 분할 (예: 99% train, 1% eval)
        # 작은 평가셋으로도 충분할 수 있습니다. 필요시 test_size 조절
        split_ratio = 0.01 
        if len(raw_dataset) * split_ratio < 1 or len(raw_dataset) * (1-split_ratio) < 1 :
            logger.warning(f"Dataset size {len(raw_dataset)} is very small. Using a single example for evaluation if possible, or skipping eval.")
            if len(raw_dataset) > 1:
                split = raw_dataset.train_test_split(test_size=1, seed=42) # 최소 1개 평가
            else: # 평가 불가
                logger.error("Dataset too small for train/test split. Consider increasing num_examples_to_take.")
                # lm_eval = None 으로 처리하거나 에러 발생
                raise ValueError("Dataset too small for reliable train/test split.")
        else:
            split = raw_dataset.train_test_split(test_size=split_ratio, seed=42)
        
        train_raw = split["train"]
        test_raw = split["test"]

        # C4와 유사한 토큰화 함수
        def tokenize_function_openwebmath(examples_batch):
            # 'text' 필드가 있다고 가정합니다. 실제 필드명 확인 필요.
            return tokenizer(examples_batch['text'], return_special_tokens_mask=True)

        logger.info("Tokenizing OpenWebMath train dataset...")
        tokenized_train = train_raw.map(
            tokenize_function_openwebmath,
            batched=True,
            remove_columns=train_raw.column_names, # 'text' 필드 등 원본 컬럼 제거
            desc="Tokenizing OpenWebMath train"
        )
        logger.info("Tokenizing OpenWebMath eval dataset...")
        tokenized_eval = test_raw.map(
            tokenize_function_openwebmath,
            batched=True,
            remove_columns=test_raw.column_names,
            desc="Tokenizing OpenWebMath eval"
        )

        # C4와 유사한 텍스트 그룹화 함수
        def group_texts_openwebmath(examples_batch):
            concatenated = {k: sum(examples_batch[k], []) for k in examples_batch.keys()}
            total_length = len(concatenated[list(examples_batch.keys())[0]]) # 첫 번째 키(예: input_ids)의 길이 사용
            
            # block_size 배수가 되도록 total_length 조정 (마지막 남는 부분 버림)
            if total_length >= args.block_size:
                total_length = (total_length // args.block_size) * args.block_size
            else: # 전체 길이가 block_size보다 작으면, 해당 배치는 비게 됨
                total_length = 0
            
            result = {
                k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
                for k, t in concatenated.items()
            }
            # 레이블 생성
            if "input_ids" in result:
                 result["labels"] = result["input_ids"].copy()
            return result

        logger.info("Grouping OpenWebMath train texts...")
        lm_train = tokenized_train.map(group_texts_openwebmath, batched=True, desc="Grouping OpenWebMath train texts")
        logger.info("Grouping OpenWebMath eval texts...")
        lm_eval = tokenized_eval.map(group_texts_openwebmath, batched=True, desc="Grouping OpenWebMath eval texts")
        
        # 그룹화 후 빈 데이터셋이 될 수 있으므로 확인
        if len(lm_train) == 0:
            logger.warning("Training dataset is empty after grouping. This might happen if total token count is less than block_size or num_examples_to_take is too small.")
        if len(lm_eval) == 0:
            logger.warning("Evaluation dataset is empty after grouping. This might happen if total token count is less than block_size or num_examples_to_take is too small for eval split.")


    else:
        raise ValueError(f"Unsupported target_dataset: {args.target_dataset}")
    
    if lm_train is None or lm_eval is None:
        raise ValueError(f"Dataset {args.target_dataset} could not be processed or is empty.")
    
    logger.info(f"Train dataset size: {len(lm_train)}")
    logger.info(f"Evaluation dataset size: {len(lm_eval)}")


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
        adam_beta1=args.adam_beta1, # Added Adam beta1
        adam_beta2=args.adam_beta2, # Added Adam beta2
        eval_strategy="steps",
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

    if args.target_task == "SLM":
        if not args.reference_model_name_or_path and args.slm_selection_strategy == "excess_loss":
            logger.error("For SLM training with 'excess_loss' strategy, --reference_model_name_or_path must be provided.")
            raise ValueError("reference_model_name_or_path is required for SLM with 'excess_loss'.")
        if args.slm_top_k_ratio == 0.0 and args.beta_rho == 0.0:
            logger.warning("Both slm_top_k_ratio and beta_rho are 0.0. Rho1Trainer will behave like a standard Trainer.")

        trainer = Rho1Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_train,
            eval_dataset=lm_eval,
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=tokenizer,
            reference_model_name_or_path=args.reference_model_name_or_path,
            beta_rho=args.beta_rho,
            slm_top_k_ratio=args.slm_top_k_ratio,
            slm_selection_strategy=args.slm_selection_strategy # Pass the strategy
        )
    elif args.target_task in ["REFERENCE", "DENSE"]:
        # For REFERENCE or DENSE training, use the standard Hugging Face Trainer
        # REFERENCE task might imply saving this model to be used as reference_model_name_or_path later
        if args.target_task == "REFERENCE":
            logger.info("Configuring Trainer for REFERENCE model training.")
        else: # DENSE
            logger.info("Configuring Trainer for DENSE model training.")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_train,
            eval_dataset=lm_eval,
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=tokenizer
        )
    else:
        # Should not happen due to choices in argparse, but as a safeguard
        raise ValueError(f"Unsupported target_task: {args.target_task}")

    logger.info(f"Starting training for task: {args.target_task}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 최종 저장 (rank 0만)
    if args.world_size == 1 or dist.get_rank() == 0:
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()