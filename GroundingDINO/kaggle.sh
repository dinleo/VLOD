export CUDA_HOME=/usr/local/cuda
pip install wandb huggingface_hub python-dotenv
pip install -e .
mkdir -p output
mkdir -p data
mkdir -p ckpt
python utils_/hf_down.py --dw_dir $DW_DIR --filename $CKPT
mv $DW_DIR/weights ckpt/weights
#python utils_/coco_add.py
#cp $ANO_PATH ./data/annotations/labels.json