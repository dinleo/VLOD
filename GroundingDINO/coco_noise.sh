#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

for noise in $(seq 0 10 70)
do
  python demo/test_noise.py \
   -c cfg.py \
   -p ckpt/weights/org_b.pth \
   --anno_path "$ANO_PATH" \
   --image_dir "$IMG_PATH" \
   --num_sample 1000 \
   --noise "$noise" \
   --up_dir "$UP_DIR" \
   --title "coco10"
done