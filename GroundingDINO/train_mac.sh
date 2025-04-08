export PYTHONPATH="/Users/leo/Code/Project/VLOD/GroundingDINO"
export TOKENIZERS_PARALLELISM=false
export WORKING="/Users/leo/Code/Project/VLOD/GroundingDINO/datasets"

CUDA_VISIBLE_DEVICES=0 \
python3 demo/train.py \
 --device cpu \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p output/weights/org_b.pth \
 --anno_path $WORKING/annotations/labels.json \
 --image_dir $COCODATA/train2017