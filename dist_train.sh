#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0
export CUDA_VISIBLE_DEVICES='0,1,2,3'

ARCH='pvt_small' # $1
GPUS=4 # $2
BATCH_SIZE=46
EPOCHS=300
OUT_PATH="./checkpoints/AliProducts_pvt_classifier_batchSize_${BATCH_SIZE}_epochs${EPOCHS}" # $3
PORT=${PORT:-29500}
# DATA_PATH='/media/Chnuphis/szq_data/imagenet'
DATA_PATH='/media/Anubis/uzair/Datasets/Products/AliProducts'

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --model $ARCH --batch-size $BATCH_SIZE --epochs $EPOCHS --data-path ${DATA_PATH} \
    --output_dir $OUT_PATH ${@:4}
