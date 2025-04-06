export CUDA_HOME=/usr/local/cuda
pip install wandb huggingface_hub python-dotenv
pip install -e .
mkdir -p groundingdino/output
python utils/hf_down.py
mv $SUB/weights output/weights