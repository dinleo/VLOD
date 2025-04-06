CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p groundingdino/weights/groundingdino_swinb_cogcoor.pth \
 --anno_path data/labels.json \
 --image_dir data/image