#!/bin/bash

MODEL_NAME_OR_PATH=/path/to/your/checkpoint/

python eval/cal_score_benchmarks_for_open_source.py --ckpt_path $MODEL_NAME_OR_PATH 
