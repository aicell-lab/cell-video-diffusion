#!/bin/bash

# Default parameters
PROMPT="A domestic scene unfolds indoors, with a parrot on a stand and a mouse-like character standing next to it, amidst a domestic setting. A lamp is knocked over, causing a sudden change in lighting and affecting the mood. The scene shifts to a maritime setting, where a sailor-like character is shown in dynamic poses near ship's wheel controls and a bell, with a view of waves and distant land through a window."
MODEL_PATH="./models/CogVideoX-5b"
LORA_PATH="./models/loras/steamboat-willie"
GENERATE_TYPE="t2v"
NUM_STEPS=30
GUIDANCE_SCALE=7.0
NUM_FRAMES=49
FPS=16
OUTPUT="./prompt3_lora.mp4"

ARGS=(
    --prompt "$PROMPT"
    --model_path "$MODEL_PATH"
    --lora_path "$LORA_PATH"
    --generate_type "$GENERATE_TYPE"
    --num_inference_steps "$NUM_STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --num_frames "$NUM_FRAMES"
    --fps "$FPS"
    --output_path "$OUTPUT"
)

# Run the generation
python -m pdb inference/cli_demo.py "${ARGS[@]}"
# python inference/cli_demo.py "${ARGS[@]}"