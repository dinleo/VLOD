CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p output/weights/ckpt_2000.pth \
 --anno_path $WORKING/annotations/labels_obj.json \
 --image_dir $COCODATA/test \
 --num_sample 100