"""
This script loads the weights of the CogVideo model and the phenotype embedder, and combines them into a single model.
"""
import torch
import os
from safetensors import safe_open
import json

BASE_DIR = "models/sft/IDR0013-FILTERED-2-t2v-special-single-final/checkpoint-1000-fp32"
with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
    config_dict = json.load(f)

from diffusers.models.transformers import CogVideoXTransformer3DModel
transformer = CogVideoXTransformer3DModel(**config_dict)
print(transformer)

from finetune.models.modules.phenotype_embedder import PhenotypeEmbedder
phenotype_embedder = PhenotypeEmbedder(
    input_dim=4,      # match your training settings
    hidden_dim=256,
    output_dim=4096,
    dropout=0.1
)
print(phenotype_embedder)

from finetune.models.modules.combined_model import CombinedTransformerWithEmbedder
combined_model = CombinedTransformerWithEmbedder(
    transformer=transformer,
    phenotype_embedder=phenotype_embedder,
    phenotype_module="single"
)
print(combined_model)

state_dict = {}

num_shards = 5
for idx in range(1, num_shards + 1):
    shard_file = os.path.join(BASE_DIR, f"model-0000{idx}-of-00005.safetensors")
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

missing, unexpected = combined_model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

combined_model.eval()
combined_model.to("cuda")

print("Model loaded and moved to GPU successfully!")
