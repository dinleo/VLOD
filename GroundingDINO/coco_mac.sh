export PYTHONPATH="/Users/leo/Code/Project/VLOD/GroundingDINO"
export TOKENIZERS_PARALLELISM=false

python3 demo/test_ap_on_coco.py \
 --device cpu \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p output/weights/org_b.pth \
 --anno_path $COCODATA/annotations/labels.json \
 --image_dir $COCODATA/train2017 \
 --num_sample 3