#!/bin/bash
ROOT_DIR="/mnt/storage/user/wangxiaodong/RLAIF-V"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd ${ROOT_DIR}/script/next

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1 #/mnt/storage/user/wangxiaodong/LLaVA-NeXT/LLaVA-NeXT-Video-7B
CONV_MODE=$2 #vicuna_v1
FRAMES=$3 #32
POOL_STRIDE=$4 #2
OVERWRITE=$5 #True
VIDEO_PATH=$6 #xU25MMA2N4aVtYay.mp4


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python video_demo.py \
    --model-path $CKPT \
    --video_path ${VIDEO_PATH} \
    --output_dir ./work_dirs/video_demo/$SAVE_DIR \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE 


