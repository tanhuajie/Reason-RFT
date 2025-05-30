#!/bin/bash
conda activate reasonrft_rl
export PYTHONPATH=$(pwd)/train
# Wandb
# export WANDB_MODE=disabled
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=vison-reason-rft
export WANDB_API_KEY="8b05b6xxxxxf224"
export WANDB_RUN_NAME=resume_finetune_qwen2vl_7b_task3_stage2_rl-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY

# Dataset
export TASK_NAME=trance-only-full
export DATASET_NAME=/path/to/your/dataset/trance-train.json
export IMAGE_PATH=/path/to/your/train_images/
export MODEL_NAME_OR_PATH=/path/to/your/checkpoints/from/stage1/
export OUTPUT_DIR=/path/to/your/checkpoints/${WANDB_RUN_NAME}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

# Debug
export DEBUG_MODE="True"
export LOG_PATH=${OUTPUT_DIR}/reward.log

torchrun --nproc_per_node=7 --nnodes=1 --master_port=29514 \
  train/stage_rl/grpo.py \
  --deepspeed scripts/train/zero3.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
  --image_path ${IMAGE_PATH} \
  --task_name ${TASK_NAME} \
  --use_vllm_for_gen true \
  --use_system_prompt false \
  --max_prompt_length 4096 \
  --max_completion_length 512 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --logging_steps 1 \
  --bf16 \
  --report_to wandb \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --max_pixels 480000 \
  --save_steps 100 \
  --num_train_epochs 1 \
  --run_name ${WANDB_RUN_NAME} \
  2>&1 | tee ${OUTPUT_DIR}/train.log
