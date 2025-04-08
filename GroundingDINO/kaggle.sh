export CUDA_HOME=/usr/local/cuda
pip install wandb huggingface_hub python-dotenv
pip install -e .
mkdir -p output
mkdir -p annotations
python utils_/hf_down.py --sub $SUB --filename $CKPT
mv $SUB/weights output/weights
python utils_/coco_add.py