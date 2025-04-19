CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c cfg.py \
 -p ckpt/weights/org_b.pth \
 --anno_path $ANO_PATH \
 --image_dir $IMG_PATH \
 --num_sample 1000