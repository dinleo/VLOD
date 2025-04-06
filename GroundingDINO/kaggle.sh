export CUDA_HOME=/usr/local/cuda
pip install -e .
mkdir groundingdino/weights
cd groundingdino/weights && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth