#!/bin/bash

set -Eeuxo pipefail

bash scripts/zsclip/zeroshot_base.sh imagenet 1 vit_b16
bash scripts/zsclip/zeroshot_new.sh imagenet 1 vit_b16

for dataset in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101; do
    bash scripts/zsclip/zeroshot_base.sh $dataset 1 vit_b16
    bash scripts/zsclip/zeroshot_new.sh $dataset 1 vit_b16
done
