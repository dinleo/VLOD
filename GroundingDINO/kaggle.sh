export CUDA_HOME=/usr/local/cuda
pip install wandb huggingface_hub python-dotenv
pip install -e .
mkdir -p output
mkdir -p annotations
python utils_/hf_down.py
mv origin/weights output/weights
python utils_/add_coco_obj.py