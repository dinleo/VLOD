#!/bin/bash

export CUDA_HOME=/usr/local/cuda
pip install wandb huggingface_hub python-dotenv
pip install -e .

mkdir -p outputs
mkdir -p intputs

python hf_down.py --dw_dir ckpt --filename ${DW_FILE1}
python hf_down.py --dw_dir ckpt --filename ${DW_FILE2}

mv ckpt/* inputs/ckpt/