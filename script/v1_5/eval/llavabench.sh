#!/bin/bash

ckpt_dir=$1 #/mnt/storage/user/wangxiaodong/RLAIF-V/llava-v1.5-7b
base_dir=./playground/data/eval/llava-bench-in-the-wild
save_name=$2 #llava-v1.5-7b
model_base=$3 # if need llava-v1.5-7b path

echo "ckpt_dir: $ckpt_dir"
echo "save_name: $save_name"
echo "model_base: $model_base"

if [ -z "$model_base" ]; then
    python llava/eval/model_vqa.py \
        --model-path ${ckpt_dir} \
        --question-file ${base_dir}/questions.jsonl \
        --image-folder ${base_dir}/images \
        --answers-file ${base_dir}/answers/${save_name}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1
else
    python llava/eval/model_vqa.py \
        --model-path $ckpt_dir \
        --model-base $model_base \
        --question-file ${base_dir}/questions.jsonl \
        --image-folder ${base_dir}/images \
        --answers-file ${base_dir}/answers/${save_name}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

fi

echo "generate answer done!"

mkdir -p ${base_dir}/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question ${base_dir}/questions.jsonl \
    --context ${base_dir}/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        ${base_dir}/answers_gpt4.jsonl \
        ${base_dir}/answers/${save_name}.jsonl \
    --output \
        ${base_dir}/reviews/${save_name}.jsonl

python llava/eval/summarize_gpt_review.py -f ${base_dir}/reviews/${save_name}.jsonl