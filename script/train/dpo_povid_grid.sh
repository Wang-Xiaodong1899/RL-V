#!/bin/bash

lr_values=("1e-7" "1e-6" "5e-6")

task_name=llava15_7b_DPO-POVID
exp_name=FT_pbs1_V8_${lr}_ZERO3


for lr in "${lr_values[@]}"
do
    model_name="llava15_7b_DPO-POVID-FT_pbs1_V8_${lr}_ZERO3"
    
    bash /root/autodl-tmp/RL-V/script/train/llava15_train_povid_dpo.sh "$lr"

    log_file="eval_${model_name}.log"
    
    bash /root/autodl-tmp/RL-V/script/v1_5/eval/pope.sh "/root/autodl-tmp/RL-V/.ckpt/${model_name}" "$model_name" > "$log_file" 2>&1
done
