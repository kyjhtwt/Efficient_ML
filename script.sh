#!/bin/sh
# SBATCH directives
#SBATCH -J efficientml	# Job name
#SBATCH -o ./out/%j.out  # Output file
##SBATCH -o ./out/%j.out
#SBATCH -t 3-00:00:00  # Run time (D-HH:MM:SS)

#### Select GPU
##SBATCH -p A100              # Partition
#SBATCH -p 3090              # Partition
##SBATCH -p A6000
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # Number of CPUs
#SBATCH --gres=gpu:4         # Number of GPUs

# 작업 제출 디렉토리로 이동
cd $SLURM_SUBMIT_DIR

# 변수 설정 (원본 스크립트에서 가져옴)
ref_root="/home/shared/efficient_ml/"
checkpoint_name="checkpoint-best-125m" 
reference_model_path="${ref_root}${checkpoint_name}"
N_GPUS=4 # SBATCH --gres=gpu 값과 일치해야 함

TARGET_TASK=("SLM" "DENSE")
MODEL_NAME="facebook/opt-2.7b"
SLM_TOP_K_RATIO=0.7 
BETA_RHO=0.0

TARGET_DATASETS=("tulu" "openhermes")

# 모듈 로드 (클러스터 환경에 맞게 수정)
echo "Loading modules..."
module purge
module load cuda/11.4.4
module load cudnn/cuda-11.4.4/8.6.0
module load nccl/cuda-11.6/2.12.12

# Python 가상 환경 활성화 (클러스터 환경에 맞게 수정)
echo "Activating Conda environment..."
echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh    # Anaconda path
conda activate efficientml

export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "Starting SLM training script..."
# 각 데이터셋에 대해 학습 실행
for dataset_name in "${TARGET_DATASETS[@]}"; do
    echo "Starting SLM training for ${MODEL_NAME} on ${dataset_name} using reference model ${reference_model_path}"

    wandb_run_name="slm-${MODEL_NAME}-${dataset_name}-ref-${checkpoint_name}"
    output_dir="./slm-output-${MODEL_NAME}-${dataset_name}"
    
    # 출력 디렉토리 생성 (이미 존재하면 무시)
    mkdir -p "${output_dir}"
    mkdir -p ./slm_output # SLURM 로그 디렉토리

    echo "Output directory: ${output_dir}"
    echo "WandB run name: ${wandb_run_name}"

    srun torchrun --standalone --nnodes=1 --nproc_per_node=${N_GPUS} main.py \
        --model_name "${MODEL_NAME}" \
        --target_dataset "${dataset_name}" \
        --target_task "${TARGET_TASK}" \
        --reference_model_name_or_path "${reference_model_path}" \
        --slm_top_k_ratio ${SLM_TOP_K_RATIO} \
        --beta_rho ${BETA_RHO} \
        --output_dir "${output_dir}" \
        --max_steps 1000 \
        --train_batch_size 8 \
        --eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --lr_scheduler_type "cosine" \
        --weight_decay 0.0 \
        --logging_steps 10 \
        --eval_steps 100 \
        --save_steps 100 \
        --save_total_limit 2 \
        --fp16 \
        --block_size 1024 \
        --slm_selection_strategy "random_loss" \
        # --resume_from_checkpoint <path_to_checkpoint_if_needed>

    echo "Finished SLM training for ${MODEL_NAME} on ${dataset_name}"
done

echo "All SLM training runs completed."

# Conda 환경 비활성화
echo "Deactivating Conda environment..."
conda deactivate

# 작업 상태 확인
squeue --job $SLURM_JOBID

echo "##### SLURM SCRIPT END #####"