CUDA_VISIBLE_DEVICES=0 \
python demo/train.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p output/weights/org_b.pth \
 --anno_path $COCODATA/annotations/labels.json \
 --image_dir $COCODATA/image