export CUDA_HOME=/usr/local/cuda
cd GroundingDINO && pip install -e .
mkdir GroundingDINO/groundingdino/weights
cd GroundingDINO/groundingdino/weights && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth