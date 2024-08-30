#!/bin/bash

set -Eeuxo pipefail

bash scripts/cocoop/base2new_train.sh imagenet 1 vit_b16_c4_ep10_batch1_ctxv1
bash scripts/cocoop/base2new_test_base.sh imagenet 1 vit_b16_c4_ep10_batch1_ctxv1
bash scripts/cocoop/base2new_test_new.sh imagenet 1 vit_b16_c4_ep10_batch1_ctxv1

for dataset in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101; do
    bash scripts/cocoop/base2new_train.sh $dataset 1 vit_b16_c4_ep10_batch1_ctxv1
    bash scripts/cocoop/base2new_test_base.sh $dataset 1 vit_b16_c4_ep10_batch1_ctxv1
    bash scripts/cocoop/base2new_test_new.sh $dataset 1 vit_b16_c4_ep10_batch1_ctxv1
done
