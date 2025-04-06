CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p output/weights/org_b.pth \
 --anno_path /kaggle/input/coco-val/data/labels.json \
 --image_dir /kaggle/input/coco-val/data/image