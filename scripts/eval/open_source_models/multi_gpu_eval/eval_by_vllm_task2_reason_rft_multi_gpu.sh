#!/bin/bash
BATCH_SIZE=32

DEVICE_IDS=(0 1 2 3 4 5 6 7)
MODEL_NAME_OR_PATH_LIST=(
    "/path/to/your/checkpoint_0/"
    "/path/to/your/checkpoint_1/"
    "/path/to/your/checkpoint_2/"
    "/path/to/your/checkpoint_3/"
    "/path/to/your/checkpoint_4/"
    "/path/to/your/checkpoint_5/"
    "/path/to/your/checkpoint_6/"
    "/path/to/your/checkpoint_7/"
)

BENCHMARK_LIST="geomath geometry3k"
STRATAGE_LIST="cot-sft cot-sft"

for i in "${!DEVICE_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_IDS[$i]} python eval/eval_by_vllm_for_open_source.py \
        --batch_size $BATCH_SIZE \
        --model_name_or_path ${MODEL_NAME_OR_PATH_LIST[$i]} \
        --benchmark_list $BENCHMARK_LIST \
        --stratage_list $STRATAGE_LIST &
done

wait
echo "All task finish."