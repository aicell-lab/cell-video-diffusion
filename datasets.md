### Live-cell imaging datasets

When going from 2D + time to 3D + time, we are doing maximum intensity projection across z. For all the below, example videos can be found at the notebooks at ./visualizations.

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
One sample 2D + time: (t, c, h, w) = (401, 1, 512, 512)

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
