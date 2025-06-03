# vedika/utils/download.py

import os
from huggingface_hub import hf_hub_download

def get_file(name: str):
    file_map = {
        "sandhi_split": "Vedika/vedika/data/sandhi_split.pth",
        "sandhi_joiner": "Vedika/vedika/data/sandhi_joiner.pth",
        "cleaned_metres": "Vedika/vedika/data/cleaned_metres.json",
    }

    if name not in file_map:
        raise ValueError(f"Unknown resource: {name}")

    return hf_hub_download(
        repo_id="tanuj437/Vedika",
        filename=file_map[name]
    )
