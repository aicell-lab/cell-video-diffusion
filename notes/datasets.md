### Live-cell imaging datasets

When going from 2D + time to 3D + time, we are doing maximum intensity projection across z. For all the below, example videos can be found at the notebooks at ./visualizations. MCA and S-BIAD1420 have segmentation masks available for each video. 

Amount of data:
```txt
[x_aleho@berzelius1 video-diffusion]$ du -sh data/raw/* | sort -h
3.7G    data/raw/mitotic_cell_atlas_v1.0.1_exampledata
6.8G    data/raw/S-BIAD1410
11G     data/raw/idr0113-bottes-opcclones
23G     data/raw/dkwh7_data
43G     data/raw/idr0115
113G    data/raw/20180215 - REMOVED
284G    data/raw/idr0067-king-yeastmeiosis - REMOVED
302G    data/raw/idr0040-aymoz-singlecell - REMOVED
326G    data/raw/S-BIAD725
462G    data/raw/mitotic_cell_atlas_v1.0.1_fulldata
4.3T    data/raw/idr0013-neumann-mitocheck
```

### MCA
Paper: https://www.nature.com/articles/s41586-018-0518-z
Data: https://www.mitocheck.org/mitotic_cell_atlas/downloads/
Description: Cells going through mitosis
We have 506 samples. 
One sample 3D + time: (t, z, c, h, w) = (40, 31, 1, 256, 256)
One sample 2D + time: (t, c, h, w) = (40, 1, 256, 256)

### S-BIAD1410
Paper: https://www.biorxiv.org/content/10.1101/2024.11.28.625889v1
Data: https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1410?query=S-BIAD1410
Description: Not sure, some kind of tissue that is healing
We have ~30 samples.
Seems to be subdatasets within each dataset.
**cardioblast**
One sample 2D + time: (t, c, h, w) = (401, 1, 512, 512)
**epidermal**
One sample 2D + time: (t, c, h, w) = (45, 1, 512, 512)
**wound margin**
One sample 2D + time: (t, c, h, w) = (121, 1, 512, 512)

### IDR-0115
Paper: https://www.nature.com/articles/s41586-022-05528-w 
Data: https://idr.openmicroscopy.org/webclient/?show=project-2301
Description: Each sample captures how labeled nucleoporins form and organize during different phases of cell division (mitosis vs. interphase).
Channels: 2. Channel 1: GFP-labelled nucleoporin; Channel 2: Silicon-rhodamine Hoechst-labelled DNA.
We have ~50 samples
One sample 3D + time: (t, z, c, h, w) = (293, 21, 2, 123, 173)
One sample 2D + time: (t, c, h, w) = (293, 2, 123, 173)

### IDR-0040
Paper: https://pubmed.ncbi.nlm.nih.gov/29695607/
Data: https://idr.openmicroscopy.org/webclient/?show=project-401
Description: Tracks how budding yeast cells respond to pheromone stimulation through the MAPK pathway, leading to distinct gene-expression programs over time. Single-cell expression reporter measurements (e.g., CFP, YFP, RFP) capture the kinetics of transcriptional induction. 
Channels: 5. 	00-BF0, 06-CFPtriple, 08-RFPtriple, 07-YFPtriple, 00-BF1, whatever this means. 
One sample 2D + time: (t, c, h, w): (20, 5, 1024, 1024)

### IDR-0067
Paper: https://elifesciences.org/articles/47156
Data: https://idr.openmicroscopy.org/webclient/?show=project-904
Description: Budding yeast cells undergoing meiosis.
WARNING: Data is ordered by figures, unclear if each image even corresponds to similar things. Channel config is different for each sample. Should probably avoid this.
We have ~100 samples
One sample 2D + time: (t, c, h, w) = (65, 2, 1024, 1024)

### IDR-0113
Paper: https://pubmed.ncbi.nlm.nih.gov/34244440/
Data: https://idr.openmicroscopy.org/webclient/?show=project-1903
WARNING: z and t are different across samples. Don't think I want to use this one for now. 

### IDR-0013
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC3108885/
Data: https://idr.openmicroscopy.org/webclient/?show=screen-1101
Description: Showing how knockdown of each gene affects cell division over two days.
The dataset looks very large.
One sample 2D + time: (t, c, h, w) = (93, 1, 1344, 1024)


### dkwh7
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC4341232/#CR17
Data: http://gigadb.org/dataset/100118
Description: In-vitro wound healing, collective cell migration. 
We have ~30 samples.
Seems to be subdatasets within each dataset. They all seem to show a similar process though.
**DKWH7**
One sample 2D + time: (t, c, h, w) = (60, 1, 1024, 1024)
**SN29**
One sample 2D + time: (t, c, h, w) = (114, 1, 1024, 1024)
**SN77**
One sample 2D + time: (t, c, h, w) = (100, 1, 1024, 1024)
