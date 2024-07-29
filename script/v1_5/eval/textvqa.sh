#!/bin/bash

ckpt_dir=$1
save_name=$2
base_dir="/mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/textvqa"
model_base=$3 # if need llava-v1.5-7b path

echo "ckpt_dir: $ckpt_dir"
echo "save_name: $save_name"
echo "model_base: $model_base"

if [ -z "$model_base" ]; then
    python llava/eval/model_vqa_loader.py \
        --model-path $ckpt_dir \
        --question-file ${base_dir}/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ${base_dir}/train_images \
        --answers-file ${base_dir}/answers/${save_name}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1
else
    python llava/eval/model_vqa_loader.py \
        --model-path $ckpt_dir \
        --model-base ${model_base} \
        --question-file ${base_dir}/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ${base_dir}/train_images \
        --answers-file ${base_dir}/answers/${save_name}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1
fi

python llava/eval/eval_textvqa.py \
    --annotation-file ${base_dir}/TextVQA_0.5.1_val.json \
    --result-file ${base_dir}/answers/${save_name}.jsonl
