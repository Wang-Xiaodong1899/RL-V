#!/bin/bash

lr_values=("1e-7" "1e-6" "5e-6")

task_name=llava15_7b_DPO_RLAIF-HIER_6k
exp_name=llava15_RLAIF-HIER-6k_pbs1_V8_${lr}_ZERO3


for lr in "${lr_values[@]}"
do
    model_name="${task_name}-${exp_name}"
    
    bash /mnt/storage/user/wangxiaodong/RLAIF-V/script/train/llava15_train_rlaif_hier.sh "$lr"

    log_file="eval_${model_name}.log"
    
    bash /mnt/storage/user/wangxiaodong/RLAIF-V/script/v1_5/eval/pope.sh "/mnt/storage/user/wangxiaodong/RLAIF-V/.ckpt/${model_name}" "$model_name" > "$log_file" 2>&1
done
