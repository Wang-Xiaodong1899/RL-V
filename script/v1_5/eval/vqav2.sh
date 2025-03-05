#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

echo $gpu_list

CHUNKS=${#GPULIST[@]}

echo $CHUNKS


SPLIT="llava_vqav2_mscoco_test-dev2015"
ckpt_dir=$1 #/data2/wangxd/models/llava-v1.5-7b
CKPT=$2 #"llava-v1.5-13b"

base_dir=./playground/data/eval/vqav2

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/model_vqa_loader.py \
        --model-path $ckpt_dir \
        --question-file ${base_dir}/$SPLIT.jsonl \
        --image-folder ${base_dir}/test2015 \
        --answers-file ${base_dir}/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${base_dir}/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${base_dir}/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python script/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

