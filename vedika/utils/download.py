from huggingface_hub import snapshot_download 
import os
import shutil

# 1. Download specific files from the repo
repo_path = snapshot_download(
    repo_id="tanuj437/Vedika",
    allow_patterns=[
        "Vedika/vedika/data/sandhi_joiner.pth",
        "Vedika/vedika/data/sandhi_split.pth",
        "Vedika/vedika/data/cleaned_metres.json",
    ]
)

# 2. Copy the needed files to your desired target folder
target_folder = "data"
os.makedirs(target_folder, exist_ok=True)

# Correct full paths inside the repo snapshot
files = [
    "Vedika/vedika/data/sandhi_joiner.pth",
    "Vedika/vedika/data/sandhi_split.pth", 
    "Vedika/vedika/data/cleaned_metres.json", 
]

for f in files:
    src = os.path.join(repo_path, f)
    dst = os.path.join(target_folder, os.path.basename(f))
    shutil.copy(src, dst)
    print(f"Copied {src} → {dst}")
