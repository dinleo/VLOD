CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p output/weights/org_b.pth \
 --anno_path $WORKING/annotations/labels_obj.json \
 --image_dir $COCODATA/train2017 \
 --num_sample 200