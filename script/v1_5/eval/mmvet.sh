#!/bin/bash

cktp_dir=$1 #/data2/wangxd/models/llava-v1.5-7b
base_dir=./playground/data/eval/mm-vet
save_name=$2 #llava-v1.5-7b

# python llava/eval/model_vqa.py \
#     --model-path ${cktp_dir} \
#     --question-file ${base_dir}/llava-mm-vet.jsonl \
#     --image-folder ${base_dir}/images \
#     --answers-file ${base_dir}/answers/${save_name}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p ${base_dir}/results

python ${base_dir}/convert_mmvet_for_eval.py \
    --src ${base_dir}/answers/${save_name}.jsonl \
    --dst ${base_dir}/results/${save_name}.json

# python ${base_dir}/mm-vet_evaluator.py \
#     --mmvet_path ${base_dir} \
#     --result_file ${base_dir}/results/${save_name}.json