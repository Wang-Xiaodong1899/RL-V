#!/bin/bash

# full models
ckpt=$1
save_name=$2
model_base=$3

echo $ckpt
echo $save_name

log_file="eval_${save_name}.log"

# cd /mnt/storage/user/wangxiaodong/RLAIF-V
# # sqa
# bash script/v1_5/eval/sqa.sh "${ckpt}" "$save_name" > "sqa_$log_file" 2>&1
# # textvqa
# bash script/v1_5/eval/textvqa.sh "${ckpt}" "$save_name" > "textvqa_$log_file" 2>&1

# cd /mnt/storage/user/wangxiaodong/debug/LLaVA/
# ## sqa
# bash scripts/v1_5/eval/sqa.sh "${ckpt}" "$save_name" > "sqa_$log_file" 2>&1
# ## textvqa
# bash scripts/v1_5/eval/textvqa.sh "${ckpt}" "$save_name" > "textvqa_$log_file" 2>&1


# SeVa POPE
cd /home/user/wangxd/RL-V/SeVa

if [ -z "$model_base" ]; then
    bash run/eval_pope_diffu500.sh "${ckpt}" "$save_name" > "/mnt/storage/user/wangxiaodong/debug/LLaVA/pope_1_$log_file" 2>&1
else
    bash run/eval_pope_diffu500.sh "${ckpt}" "$save_name" "$model_base" > "/mnt/storage/user/wangxiaodong/debug/LLaVA/pope_1_$log_file" 2>&1
fi