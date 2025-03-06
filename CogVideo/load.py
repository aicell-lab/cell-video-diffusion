import torch
import os
from safetensors import safe_open
import json

# 1. Load config
BASE_DIR = "/proj/aicell/users/x_aleho/video-diffusion/CogVideo/models/sft/IDR0013-FILTERED-2-t2v-special-single-final/checkpoint-1000-fp32"
with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
    config_dict = json.load(f)

# 2. Instantiate the base transformer
from diffusers.models.transformers import CogVideoXTransformer3DModel
transformer = CogVideoXTransformer3DModel(**config_dict)
print(transformer)

# 3. Instantiate your phenotype embedder (if you used single or multi)
from finetune.models.modules.phenotype_embedder import PhenotypeEmbedder
phenotype_embedder = PhenotypeEmbedder(
    input_dim=4,      # match your training settings
    hidden_dim=256,
    output_dim=4096,
    dropout=0.1
)
print(phenotype_embedder)

# 4. Wrap them in the same structure as training
from finetune.models.modules.combined_model import CombinedTransformerWithEmbedder
combined_model = CombinedTransformerWithEmbedder(
    transformer=transformer,
    phenotype_embedder=phenotype_embedder,
    phenotype_module="single"  # or "multi", whichever you used
)
print(combined_model)

# 5. Load the safetensors weights
state_dict = {}

num_shards = 5
for idx in range(1, num_shards + 1):
    shard_file = os.path.join(BASE_DIR, f"model-0000{idx}-of-00005.safetensors")
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

# 6. Load into the combined model
missing, unexpected = combined_model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

combined_model.eval()
combined_model.to("cuda")

print("Model loaded and moved to GPU successfully!")
