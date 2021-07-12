#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0
export CUDA_VISIBLE_DEVICES='0,2,3,4,5,6'

ARCH='pvt_large' # $1
GPUS=6 # $2
BATCH_SIZE=150
EPOCHS=300
# OUT_PATH="./checkpoints/AliProducts_pvt_classifier_batchSize_${BATCH_SIZE}_epochs${EPOCHS}" # $3
OUT_PATH="./checkpoints/targets_${ARCH}_classifier_batchSize_${BATCH_SIZE}_epochs${EPOCHS}_codeHyperParams_AliUnclean_plus_bottles" # $3
PORT=${PORT:-29500}
# DATA_PATH='/media/Chnuphis/szq_data/imagenet'
# DATA_PATH='/media/Anubis/uzair/Datasets/Products/AliProducts'
DATA_PATH='/media/not_using_this_path'

python -W ignore -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --model $ARCH --batch-size $BATCH_SIZE --epochs $EPOCHS --data-path ${DATA_PATH} \
    --output_dir $OUT_PATH ${@:4} 
