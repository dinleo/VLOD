import argparse
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
HUG = os.getenv('HUG')
api = HfApi()

# 인자 파서 설정
parser = argparse.ArgumentParser(description="Download a specific file from Hugging Face Hub")
parser.add_argument('--dw_dir', type=str, default="origin", help='Subfolder path under the repo')
parser.add_argument('--filename', type=str, default="org_b", help='Name of the file to download')

args = parser.parse_args()

# 다운로드 실행
downloaded_path = api.hf_hub_download(
    local_dir='./',
    repo_id="dinleo11/Aqua",
    subfolder=f"{args.dw_dir}",
    filename=f"{args.filename}",
    repo_type="model",
    token=HUG,
)

print(f"Downloaded to: {downloaded_path}")
