#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=ZeroshotCLIP
DATASET=$1
SEED=$2
CFG=$3  # rn50, rn101, vit_b32 or vit_b16
SUB=base

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB}