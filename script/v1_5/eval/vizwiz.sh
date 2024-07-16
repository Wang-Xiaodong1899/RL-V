#!/bin/bash

ckpt_dir=$1
save_name=$2
base_dir="./playground/data/eval/vizwiz"

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt_dir \
    --question-file ${base_dir}/llava_test.jsonl \
    --image-folder ${base_dir}/test \
    --answers-file ${base_dir}/answers/${save_name}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python script/convert_vizwiz_for_submission.py \
    --annotation-file ${base_dir}/llava_test.jsonl \
    --result-file ${base_dir}/answers/${save_name}.jsonl \
    --result-upload-file ${base_dir}/answers_upload/${save_name}.json
