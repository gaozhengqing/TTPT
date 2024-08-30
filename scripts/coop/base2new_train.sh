#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=$3
SHOTS=16

CTX_INIT=$4
CLASS_TOKEN_POSITION=$5

DIR=my_outputv2/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.COOP.CTX_INIT "${CTX_INIT}" \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CLASS_TOKEN_POSITION}
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.COOP.CTX_INIT "${CTX_INIT}" \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CLASS_TOKEN_POSITION}
fi