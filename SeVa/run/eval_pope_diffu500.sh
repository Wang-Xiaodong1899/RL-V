MODEL_VERSION=$1
save_name=$2
model_base=$3 # /root/autodl-fs/llava-v1.5-7b

# MODEL_VERSION=/root/autodl-tmp/RL-V/seva-7b-diffu500

if [ -z "$model_base" ]; then
    torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
        --coco_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --pope_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --model-path ${MODEL_VERSION} \
        --save_dir ./seva/pope_result/${save_name} \
        --set random
else
    torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
        --coco_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --pope_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --model-path ${MODEL_VERSION} \
        --model-base ${model_base} \
        --save_dir ./seva/pope_result/${save_name} \
        --set random
fi

if [ -z "$model_base" ]; then
    torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
        --coco_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --pope_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --model-path ${MODEL_VERSION} \
        --save_dir ./seva/pope_result/${save_name} \
        --set popular
else
    torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
        --coco_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --pope_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --model-path ${MODEL_VERSION} \
        --model-base ${model_base} \
        --save_dir ./seva/pope_result/${save_name} \
        --set popular
fi

if [ -z "$model_base" ]; then
    torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
    --coco_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
    --pope_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
    --model-path ${MODEL_VERSION} \
    --save_dir ./seva/pope_result/${save_name} \
    --set adv
else
    torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
        --coco_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --pope_path /mnt/storage/user/wangxiaodong/debug/LLaVA/playground/data/eval/pope/ \
        --model-path ${MODEL_VERSION} \
        --model-base ${model_base} \
        --save_dir ./seva/pope_result/${save_name} \
        --set adv
fi

# convert
python /root/autodl-tmp/RL-V/convert_pope_answer.py --dirs ./seva/pope_result/${save_name}

cd /mnt/storage/user/wangxiaodong/debug/LLaVA

base_dir=./playground/data/eval/pope

python llava/eval/eval_pope.py \
    --annotation-dir ${base_dir}/coco \
    --question-file ${base_dir}/llava_pope_test.jsonl \
    --result-file /root/autodl-tmp/RL-V/SeVa/seva/pope_result/${save_name}/pope_all.jsonl

# python seva/pope_calculate.py --path ./seva/pope_result/${MODEL_VERSION}
