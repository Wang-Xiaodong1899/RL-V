#!/bin/bash

# lr_values=("1e-7" "1e-6" "5e-6")

lr=$1

task_name=llava15_7b_DPO-POVID
exp_name=FT_pbs1_V8_${lr}_ZERO3

model_name="${task_name}-${exp_name}"

log_file="eval_${model_name}.log"
    
bash /workspace/wxd/RL-V/script/v1_5/eval/pope.sh "/workspace/wxd/RL-V/.ckpt/${model_name}/checkpoints/" "$model_name" > "$log_file" 2>&1
