#!/usr/bin/env python3

import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

hf_token = os.environ["HF_TOKEN"]

models_dir = Path.cwd() / "models"
output_dir = models_dir / "Llama-3.1-8B-Instruct"

models_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_ID}")
print(f"Destination: {output_dir}")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=output_dir,
    token=hf_token,
)

print("Download complete.")