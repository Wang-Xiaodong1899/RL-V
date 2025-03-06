#!/bin/bash

ckpt_dir=$1 #/volsparse3/wxd/models/llava-v1.5-7b
save_name=$2 #llava-v1.5-7b
model_base=$3 # if need llava-v1.5-7b path

base_dir=/mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/scienceqa

echo "ckpt_dir: $ckpt_dir"
echo "save_name: $save_name"
echo "model_base: $model_base"

if [ -z "$model_base" ]; then
    # not provide
    python llava/eval/model_vqa_science.py \
        --model-path $ckpt_dir \
        --question-file ${base_dir}1/llava_test_CQM-A.json \
        --image-folder ${base_dir}/images/test \
        --answers-file ${base_dir}/answers/${save_name}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1
else
    python llava/eval/model_vqa_science.py \
        --model-path $ckpt_dir \
        --model-base $model_base \
        --question-file ${base_dir}/llava_test_CQM-A.json \
        --image-folder ${base_dir}/images/test \
        --answers-file ${base_dir}/answers/${save_name}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1
fi
echo "generate answer done!"

python llava/eval/eval_science_qa.py \
    --base-dir $base_dir \
    --result-file ${base_dir}/answers/${save_name}.jsonl \
    --output-file ${base_dir}/answers/${save_name}_output.jsonl \
    --output-result ${base_dir}/answers/${save_name}_result.json
