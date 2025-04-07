export CUDA_HOME=/usr/local/cuda
pip install wandb huggingface_hub python-dotenv
pip install -e .
mkdir -p output/weights
python utils_/hf_down.py
mv org/weights output/weights
python utils_/add_coco_obj.py