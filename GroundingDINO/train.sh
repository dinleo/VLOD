CUDA_VISIBLE_DEVICES=0 \
python demo/train.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p output/weights/org_b.pth \
 --anno_path $WORKING/annotations/labels_only.json \
 --image_dir $COCODATA/train2017