#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=TTPT
DATASET=$1
SEED=$2
CFG=$3  # rn50, rn101, vit_b32 or vit_b16
SHOTS=$4
LOADEP=$5
SUB=new
CTX_INIT=$6
CLASS_TOKEN_POSITION=$7

MODEL_DIR=my_outputv2/base2new/train_base/${DATASET}/shots_${SHOTS}/CoOp/${CFG}/seed${SEED}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir my_outputv2_tau_1/${TRAINER}/${CFG}/${DATASET} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB} \
TRAINER.COOP.CTX_INIT "${CTX_INIT}" \
TRAINER.COOP.CLASS_TOKEN_POSITION ${CLASS_TOKEN_POSITION}
