conda activate reasonrft_sft

export PYTHONPATH=$(pwd)/train/stage_sft

export WANDB_MODE=offline
export ACCELERATE_CPU_AFFINITY=1

export IMAGE_DIR=/path/to/your/train_images/
export PRETRAIN_MODEL_PATH=/path/to/your/pretrain_model/Qwen2-VL-7B-Instruct
export OUTPUT_PATH=/path/to/your/checkpoints/qwen2vl_7b_task2_ans_sft
export DATASET=geo_math_sft

if [ ! -d "$OUTPUT_PATH" ]; then
  mkdir "$OUTPUT_PATH"
fi

torchrun --nproc_per_node=8 --nnodes=1 --master_port=29514 \
  train/stage_sft/train.py \
  --deepspeed scripts/train/zero3.json \
  --stage sft \
  --do_train \
  --model_name_or_path $PRETRAIN_MODEL_PATH \
  --dataset $DATASET \
  --image_dir $IMAGE_DIR \
  --template qwen2_vl \
  --finetuning_type full \
  --output_dir $OUTPUT_PATH \
  --overwrite_cache \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --ddp_timeout 90000 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --cutoff_len 4096 \
  --save_steps 200 \
  --plot_loss \
  --num_train_epochs 1 \
  --bf16 \
  2>&1 | tee ${OUTPUT_DIR}/train.log