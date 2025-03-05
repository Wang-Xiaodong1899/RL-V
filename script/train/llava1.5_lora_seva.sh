MODEL_VERSION=RLAIFV-HIER-6k_r1024_a2048_pbs4_V8_1e-7

OCR_DPO_DATA=/home/user/wangxd/RL-V/seva_data/ocrvqa_dpo_8k_diffusion_step500.json
TEXT_DPO_DATA=/home/user/wangxd/RL-V/seva_data/textvqa_dpo_8k_filter_diffusion_step500.json


project_name=LLaVA-LoRA
export WANDB_PROJECT=${project_name}

deepspeed muffin/train/train_dpo_lora.py \
    --lora_enable True --lora_r 1024 --lora_alpha 2048 --mm_projector_lr 0 \
    --deepspeed ./script/zero3.json \
    --model_name_or_path /data2/wangxd/models/llava-v1.5-7b \
    --version v1 \
    --ocr_data_path ${OCR_DPO_DATA} \
    --ocr_image_path /home/user/wangxd/RL-V/ocr-qa/ocr_vqa/images/ \
    --textvqa_data_path ${TEXT_DPO_DATA} \
    --textvqa_image_path /home/user/wangxd/RL-V/playground/data/eval/textvqa/train_images/ \
    --vision_tower /mnt/storage/user/wangxiaodong/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/ \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir .ckpt_lora/${MODEL_VERSION} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${MODEL_VERSION} \
    --beta 0.1


