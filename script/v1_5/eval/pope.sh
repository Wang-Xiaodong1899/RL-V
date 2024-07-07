#!/bin/bash

cktp_dir=$1 #/mnt/storage/user/wangxiaodong/RLAIF-V/llava-v1.5-7b
base_dir=./playground/data/eval/pope
save_name=$2 #llava-v1.5-7b

python llava/eval/model_vqa_loader.py \
    --model-path ${cktp_dir} \
    --question-file ${base_dir}/llava_pope_test.jsonl \
    --image-folder ${base_dir}/val2014 \
    --answers-file ${base_dir}/answers/${save_name}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ${base_dir}/coco \
    --question-file ${base_dir}/llava_pope_test.jsonl \
    --result-file ${base_dir}/answers/${save_name}.jsonl
