#!/usr/bin/env bash

# A list of plate directories you want to process
PLATES=(
    "/proj/aicell/users/x_aleho/video-diffusion/data/raw/idr0013/LT0001_02--ex2005_11_16--sp2005_02_17--tt17--c3/hdf5"
    "/proj/aicell/users/x_aleho/video-diffusion/data/raw/idr0013/LT0001_09--ex2005_11_16--sp2005_02_17--tt17--c5/hdf5"
    "/proj/aicell/users/x_aleho/video-diffusion/data/raw/idr0013/LT0001_12--ex2005_05_13--sp2005_02_17--tt17--c4"
    "/proj/aicell/users/x_aleho/video-diffusion/data/raw/idr0013/LT0002_02--ex2005_10_26--sp2005_03_04--tt173--c5/hdf5"
    "/proj/aicell/users/x_aleho/video-diffusion/data/raw/idr0013/LT0002_24--ex2005_05_06--sp2005_03_04--tt163--c5/hdf5"
)

# Frames per second for MP4
FPS=10

# The Python script that does the conversion
CONVERTER_SCRIPT="./process_ch5_to_mp4.py"

# Number of parallel jobs
N_JOBS=5

# Now we call "parallel"
parallel -j "${N_JOBS}" python "${CONVERTER_SCRIPT}" {} "${FPS}" ::: "${PLATES[@]}"