#!/bin/bash

set -Eeuxo pipefail

bash scripts/coop/base2new_train_imagenet.sh imagenet 1 vit_b16_ep50_ctxv1 "a photo of a" end
bash scripts/coop/base2new_test_base_imagenet.sh imagenet 1 vit_b16_ep50_ctxv1 "a photo of a" end
bash scripts/coop/base2new_test_new_imagenet.sh imagenet 1 vit_b16_ep50_ctxv1 "a photo of a" end

for dataset in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101; do
    bash scripts/coop/base2new_train.sh $dataset 1 vit_b16_ctxv1 "a photo of a" end
    bash scripts/coop/base2new_test_base.sh $dataset 1 vit_b16_ctxv1 "a photo of a" end
    bash scripts/coop/base2new_test_new.sh $dataset 1 vit_b16_ctxv1 "a photo of a" end
done
