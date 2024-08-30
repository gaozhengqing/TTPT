#!/bin/bash

set -Eeuxo pipefail

bash scripts/coop/base2new_train_imagenet.sh imagenet 1 vit_b32_ep50_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh imagenet 1 vit_b32_ep50_ctxv1 16 50 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh imagenet 1 vit_b32_ep50_ctxv1 16 50 "a photo of a" end

bash scripts/coop/base2new_train.sh caltech101 1 vit_b16_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh caltech101 1 vit_b16_ctxv1 16 200 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh caltech101 1 vit_b16_ctxv1 16 200 "a photo of a" end

bash scripts/coop/base2new_train.sh oxford_pets 1 vit_b16_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh oxford_pets 1 vit_b16_ctxv1 16 200 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh oxford_pets 1 vit_b16_ctxv1 16 200 "a photo of a" end

bash scripts/coop/base2new_train.sh stanford_cars 1 vit_b16_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh stanford_cars 1 vit_b16_ctxv1 16 200 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh stanford_cars 1 vit_b16_ctxv1 16 200 "a photo of a" end

bash scripts/coop/base2new_train.sh oxford_flowers 1 vit_b16_ctxv1 "a photo of a , a type of flower" middle
bash scripts/ttpt/base2new_test_base.sh oxford_flowers 1 vit_b16_ctxv1 16 200 "a photo of a , a type of flower" middle
bash scripts/ttpt/base2new_test_new.sh oxford_flowers 1 vit_b16_ctxv1 16 200 "a photo of a , a type of flower" middle

bash scripts/coop/base2new_train.sh food101 1 vit_b16_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh food101 1 vit_b16_ctxv1 16 200 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh food101 1 vit_b16_ctxv1 16 200 "a photo of a" end

bash scripts/coop/base2new_train.sh fgvc_aircraft 1 vit_b16_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh fgvc_aircraft 1 vit_b16_ctxv1 16 200 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh fgvc_aircraft 1 vit_b16_ctxv1 16 200 "a photo of a" end

bash scripts/coop/base2new_train.sh sun397 1 vit_b16_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh sun397 1 vit_b16_ctxv1 16 200 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh sun397 1 vit_b16_ctxv1 16 200 "a photo of a" end

bash scripts/coop/base2new_train.sh dtd 1 vit_b16_ctxv1 "a photo of a" end
bash scripts/ttpt/base2new_test_base.sh dtd 1 vit_b16_ctxv1 16 200 "a photo of a" end
bash scripts/ttpt/base2new_test_new.sh dtd 1 vit_b16_ctxv1 16 200 "a photo of a" end

bash scripts/coop/base2new_train.sh eurosat 1 vit_b16_ctxv1 "a centered satellite photo of" end
bash scripts/ttpt/base2new_test_base.sh eurosat 1 vit_b16_ctxv1 16 200 "a centered satellite photo of" end
bash scripts/ttpt/base2new_test_new.sh eurosat 1 vit_b16_ctxv1 16 200 "a centered satellite photo of" end

bash scripts/coop/base2new_train.sh ucf101 1 vit_b16_ctxv1 "a photo of a person doing" end
bash scripts/ttpt/base2new_test_base.sh ucf101 1 vit_b16_ctxv1 16 200 "a photo of a person doing" end
bash scripts/ttpt/base2new_test_new.sh ucf101 1 vit_b16_ctxv1 16 200 "a photo of a person doing" end
