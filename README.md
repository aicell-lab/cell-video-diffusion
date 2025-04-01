# Cell Video Diffusion

## Environment Setup

1. Clone repo:
```bash
git clone https://github.com/aicell-lab/cell-video-diffusion.git
```
2. Create the environment:
```bash
mamba env create -f environment.yaml
```
3. Activate the environment:
```bash
mamba activate cogvideo
```
4. Install diffusers stuff:
```bash
cd CogVideo/diffusers
pip install -e .
```

## Data

This directory is obviously not pushed to github, but lays out how the code assumes that the data is structured. The /data directory contains all the data for the project, organized into several subdirectories:

- **`/data/raw`**: Contains unprocessed files from resources like IDR
- **`/data/processed`**: Contains data that has been processed into other formats, primarily MP4
- **`/data/ready`**: Contains the prepared folders ready for training video diffusion models

For this study we used the IDR0013 (https://pmc.ncbi.nlm.nih.gov/articles/PMC3108885/) dataset, downloading it with lftp from IDR (https://idr.openmicroscopy.org/).


### Data Processing
Can be found at /scripts. All the code here is assuming that we the IDR0013 (https://pmc.ncbi.nlm.nih.gov/articles/PMC3108885/) dataset. The scripts/i2v and scripts/t2v both directory contain ordered scripts that should all just be run in order. Each script has a docstring at the top that explains how it should be run. 

# Model Training

## CogVideo

The base model we are using is CogVideoX (https://github.com/THUDM/CogVideo). It uses diffusers and accelerate for training. Diffusers was installed via git outside of conda, and all the diffusers code we are using is contained in this repo and pushed to github.  Descriptions of some of the directories

### CogVideo/finetune/
Contains scripts for fine-tuning pre-trained video diffusion models on our cell video datasets. Notably, we have several bash files for starting training, for example `train_ddp_i2v.sh` which is if for doing LoRA fine-tuning, and `train_zero_i2v.sh` for doing full fine-tuning (called SFT by the CogVideo folks). The GPU memory requirements are layed out in detail at https://github.com/THUDM/CogVideo/tree/main/finetune, but to summarize; you will get OOM if using a 40gb RAM GPU both when doing LoRA and SFT, so use a bigger GPU. SFT requires usage of DeepSpeed ZeRO, and the configs can be found here at accelerate_config{n}.yaml where n says how many GPUs we are using. 

### CogVideo/inference/
Contains scripts for running the models at inference. Use the bash scripts present here to get some videos, the main script that they use is cli_demo.py. At final_evals/ most of the scripts that was used for the generations used in the paper can be found, along with two scripts to generate videos in batches. 

## Evaluation
The base files are:
- video_utils.py
- segmentation.py
- split_videos_file.py
- create_masks.py
- division_analysis.py
- morphology_analysis.py
- movement_analysis.py
- division_analysis.py

These are then used by the following scripts to create csv files with statistics at evaluation/results:
- compute_{morphology, movement, proliferation}_distribution.py
- run_compute_{morphology, movement, proliferation}_distributions.sh

To then get to the final evaluation numbers, the notebooks at evaluation/notebooks are used. 
