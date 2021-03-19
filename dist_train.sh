#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0
export CUDA_VISIBLE_DEVICES='0,1,2,3'

ARCH='pvt_small' # $1
GPUS=4 # $2
OUT_PATH='./checkpoints' # $3
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --model $ARCH --batch-size 128 --epochs 300 --data-path /path/to/imagenet \
    --output_dir $OUT_PATH ${@:4}
