Use checkpoint 1200 for all. 

### 1. Unconditional Model
	•	No prompt at inference (empty string).
	•	Generate 50 videos total, each with a different random seed.
	•	Result: One unconditional distribution of 50 videos.

### 2. Text-Conditioned Model
For each of the four phenotypes – proliferation (pr), cell death (cd), migration speed (ms), and cell count (cc) – you will generate:
	1.	HIGH label:
	•	50 videos (50 different random seeds).
	•	E.g. use a textual prompt like “Time-lapse microscopy video of cells. The cells divide often … etc.”
	2.	LOW label:
	•	50 videos (50 different random seeds).
	•	E.g. “Time-lapse microscopy video of cells. The cells rarely divide … etc.”

That means for each phenotype, you end up with 100 videos (HIGH + LOW).
	•	Total (across pr, cd, ms, cc): 400 videos.
We end up with 8 distributions: pr-HIGH, pr-LOW, cd-HIGH, cd-LOW, ms-HIGH, ms-LOW, cc-HIGH, cc-LOW.

### 3. Single-Token Numeric Model
Similarly, for each of the four dimensions – pr, cd, ms, cc – define a numeric “HIGH” value (e.g. 0.8) and a “LOW” value (e.g. 0.2), then:
	1.	HIGH numeric setting:
	•	50 videos (50 seeds).
	2.	LOW numeric setting:
	•	50 videos (50 seeds).

That’s another 100 videos per phenotype:
	•	Total (across pr, cd, ms, cc): 400 videos.
We end up with another 8 distributions. 

## Summary of Runs
	•	Unconditional: 50
	•	Text: 400 (100 per phenotype)
	•	Numeric: 400 (100 per phenotype)

Overall total: 850 generated videos (assuming no additional combos).

## Comparison with real distribution
We have 300 test videos. Partition according to the high and low values. 

rsync -avz --progress -e "ssh -p 20636 -i ~/.ssh/id_ed25519" /proj/aicell/users/x_aleho/video-diffusion/CogVideo/models/sft/t2v-uncond/checkpoint-1000-fp32 root@157.157.221.29:/workspace/cell-video-diffusion/CogVideo/models/sft/t2v-uncond