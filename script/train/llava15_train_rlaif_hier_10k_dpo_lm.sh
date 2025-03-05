export PYTHONPATH=$PYTHONPATH:`realpath .`

task_name=llava15_7b_DPO-LM_RLAIF-HIER_10k
exp_name=llava15-pbs1_V8_step-3750_ZERO3_sft-weight-0.01

export WANDB_PROJECT=$task_name
export SFT_weight=0.01

# zero-2 for process logps
# zero-3 for training

deepspeed ./muffin/train/train_llava15.py \
    --deepspeed ./script/zero3.json  \
    --model_name_or_path /data2/wangxd/models/llava-v1.5-7b \
    --data_dir /home/user/wangxd/RL-V/RLAIF-HIER-Dataset-10k_logps \
    --is_multimodal True \
    --image_folder not_used \
    --vision_tower /mnt/storage/user/wangxiaodong/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/ \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fully_tune True \
    --image_aspect_ratio pad \
    --bf16 False \
    --fp16 True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --output_dir .ckpt/$task_name-$exp_name/checkpoints \
    --max_steps 3750 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1250 \
    --save_total_limit 5 \
    --data_source_names '' \
    --data_source_weights 1 \
    --learning_rate 5e-7 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --logging_dir .ckpt/$task_name-$exp_name/log \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to wandb \
    --run_name $exp_name \
    --dataloader_num_workers 16 \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta 0.1