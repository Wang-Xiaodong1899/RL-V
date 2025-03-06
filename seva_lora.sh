HTTPS_PROXY=http://fvgroup:48423590@10.54.0.93:3128
bash script/train/llava1.5_lora_seva.sh
echo "training done"

bash script/v1_5/eval/pope.sh /root/autodl-tmp/RL-V/.ckpt_lora/RLAIFV-HIER-6k_r1024_a2048_pbs4_V8_1e-7/ RLAIFV-HIER-6k_r1024_a2048_pbs4_V8_1e-7 /root/autodl-fs/llava-v1.5-7b