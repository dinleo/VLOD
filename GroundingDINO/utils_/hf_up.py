from dotenv import load_dotenv
import os
from huggingface_hub import HfApi

load_dotenv()
HUG = os.getenv('HUG')
api = HfApi()

def upload(up_dir):
    if up_dir != "":
        print("Uploading Huggingface Hub:", up_dir)
        api.upload_folder(
            folder_path='./output', # 폴더 안 내용물만 들어간다.
            repo_id="dinleo11/VLOD", # 레포 주소
            path_in_repo=up_dir, # 레포 내 저장할 폴더
            repo_type="model",
            token=HUG
        )
