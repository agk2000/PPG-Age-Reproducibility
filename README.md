# PPG Age Reproducibility (UK Biobank)

This repo contains the analysis code used in our paper on reproducibility of age prediction from photoplethysmography (PPG) signals, using UK Biobank as the primary public dataset. It includes scripts/notebooks for:
- ASI (arterial stiffness index) vs age visualization and an ASI-only age prediction baseline
- mean-age baseline
- deep learning baselines (SMoLK, ResNet)
- PaCMAP plots of learned embeddings
- figure-generation notebooks

We do not redistribute UK Biobank data. You must have approved access and provide your own exported files.

## What’s in this repo

- `asi_prediction.py`  
  5-fold cross-validation baseline predicting age from ASI only (simple linear fit). Writes a CSV of per-fold metrics.

- `ASI_plot.ipynb`  
  Plots ASI vs age and summary distributions.

- `MeanEstimating.ipynb`  
  5-fold cross-validation “mean predictor” baseline (predict train-fold mean age for all validation samples).

- `SMoLK.py`  
  SMoLK model definition / training code used for age prediction experiments.

- `resnet.py`  
  1D ResNet model definition / training code used for age prediction experiments.

- `PacMAP.ipynb`  
  PaCMAP projections of learned embeddings for visualization.

- `figure1.ipynb`  
  Code to generate Figure 1-style panels (PPG examples / groupings used in the manuscript).

## Quickstart

### UK Biobank access (required)

To run these analyses you need UK Biobank approval for the relevant fields and an export of those fields to a CSV accessible within your compute environment (sDNAnexus).

High level steps:
1. Apply for UK Biobank access and obtain approval for your application.
2. Ensure your approved field list includes the variables used here (age, ASI, and PPG waveform field).
3. Export a table with those fields into a CSV (called `data.csv` by default in this repo), then run the scripts/notebooks.

This repo assumes you already have a CSV extracted from UK Biobank. It does not include code to request data access or to bypass UK Biobank’s access controls.

### Expected `data.csv` format

Most scripts/notebooks assume a CSV named `data.csv` in the working directory. If your file has a different name or lives elsewhere, either:
- rename it to `data.csv`, or
- edit the first cell / the `pd.read_csv(...)` line to point to your path.

The code expects these columns:

- `p21003_i0` : age (years)
- `p21021_i0` : arterial stiffness index (ASI)
- `p4205_i0`  : PPG waveform encoding (a string that needs parsing in some analyses)

Notes:
- PPG-based analyses require `p4205_i0` and assume each entry is a delimited string that can be parsed into a numeric waveform.
