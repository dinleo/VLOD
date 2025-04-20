CUDA_VISIBLE_DEVICES=0 \
python demo/train.py \
 -c cfg.py \
 -p ckpt/weights/org_b.pth \
 --anno_path "$ANO_PATH" \
 --image_dir "$IMG_PATH" \
 --up_dir "$UP_DIR"