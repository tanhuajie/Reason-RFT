#!/bin/bash

DEVICE_ID=0
BATCH_SIZE=32

MODEL_NAME_OR_PATH=/path/to/your/checkpoint/
BENCHMARK_LIST="trance trance-left trance-right clevr-math super-clevr geomath geometry3k"
STRATAGE_LIST="cot-sft cot-sft cot-sft cot-sft cot-sft cot-sft cot-sft"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval/eval_by_vllm_for_open_source.py \
    --batch_size $BATCH_SIZE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --benchmark_list $BENCHMARK_LIST \
    --stratage_list $STRATAGE_LIST