from dotenv import load_dotenv
import os
from huggingface_hub import HfApi

load_dotenv()
HUG = os.getenv('HUG')
sub = os.getenv('SUB')
api = HfApi()

print(api.hf_hub_download(
    local_dir='',
    repo_id="dinleo11/VLOD",
    subfolder=f"{sub}/weights",
    filename="org_b.pth",
    repo_type="model",
    token=HUG,
))
