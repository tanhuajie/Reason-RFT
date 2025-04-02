#!/bin/bash
python eval/cal_score_benchmarks_for_close_source.py --task_name clevr-math --model_type gpt4o
python eval/cal_score_benchmarks_for_close_source.py --task_name super-clevr --model_type gpt4o
python eval/cal_score_benchmarks_for_close_source.py --task_name geomath --model_type gpt4o
python eval/cal_score_benchmarks_for_close_source.py --task_name geometry3k --model_type gpt4o
python eval/cal_score_benchmarks_for_close_source.py --task_name trance --model_type gpt4o
python eval/cal_score_benchmarks_for_close_source.py --task_name trance-left --model_type gpt4o
python eval/cal_score_benchmarks_for_close_source.py --task_name trance-right --model_type gpt4o