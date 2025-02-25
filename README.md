# Cell Video Diffusion

## Data Organization

The project data is organized in the following structure:

### `/data`
This directory contains all the data for the project, organized into several subdirectories:

- **`/data/raw`**: Contains unprocessed files from resources like IDR
- **`/data/processed`**: Contains data that has been processed into other formats, primarily MP4
- **`/data/ready`**: Contains the prepared folders ready for training video diffusion models

## Environment Setup

This project uses a conda environment defined in `environment.yaml`. To set up the environment:

1. Create the environment:
   ```bash
   mamba env create -f environment.yaml
   ```
2. Activate the environment:
   ```bash
   mamba activate cogvideo
   ```

## Data Processing

### IDR0013 Dataset

The `scripts/idr0013` directory contains scripts for processing the IDR0013 dataset:

1. **`01-process_ch5_to_mp4.py`**: Converts CellH5 files to MP4 videos

2. **`02-score-videos.py`**: Analyzes videos to compute proliferation scores

3. **`03-prepare-idr0013.py`**: Prepares training and validation datasets with binned proliferation scores

## Model Training

### CogVideo

The `CogVideo` directory contains the code for training and fine-tuning video diffusion models:

- **`finetune/`**: Contains scripts for fine-tuning pre-trained video diffusion models on our cell video datasets