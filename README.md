<p align="center">
  <img src="images/banner.png" alt="Banner" width="100%">
</p>

<h1 align="center"> TemporalKV: A Sub-4-bit KV-Cache Compression Framework </h1>

<p align="center">
  <b>Lightweight. Efficient. Accurate.</b>  
</p>

## Informations

**Author:** Swann Bessa
**Degree:** Master's Thesis  
**University:** University of Oxford
**Year:** 2025

---

## Overview

This repository contains the implementation of **TemporalKV**, a framework for **sub-4-bit KV-Cache compression**.  
It enables memory-efficient inference for large language models while maintaining strong accuracy.

---

## Repository Structure

- **`write_centroids_bis.py`** → Performs centroid calibration using collected activations.  
- **`test_model2.py`** → Integrates the compression framework in the LLM pipeline; used for perplexity & accuracy evaluation.  
- **`cache_utils.py`** → Custom low-precision cache implementation with outlier removal.  
- **`run-fisher2.py`** → Collects activations, gradients, and statistics (quantiles, normalization constants).  
- **`datautils.py`** → Utility functions for datasets and loaders. (from [KVQuant's repository](https://github.com/SqueezeAILab/KVQuant))
- **`results.csv`** → Experimental results (accuracy & perplexity).  
- **`jobs/`** → Shell scripts to launch experiments.  
- **`experiment_notebooks/`** → Jupyter notebooks for preliminary experiments.  
- **`fast_pytorch_kmeans/`** → Modified GPU K-Means (forked from [fast_pytorch_kmeans](https://github.com/DeMoriarty/fast_pytorch_kmeans)).  
- **`mses/`** → Mean squared error results as `.npy` arrays.
- **`transformers.zip`** → Zipped custom Transformers library  


## Running the code

```bash
# Clone the repository
git clone https://github.com/bessa75/TemporalKV
cd TemporalKV

# Install dependencies
conda create -n temp_env python=3.11.13 -y
conda activate temp_env
pip install -r requirements.txt

#Unzip custom Transformers library and install it
unzip transformers.zip
cd transformers
pip install .
cd ..

# Collect activations
Use jobs/job_collect.sh

# Run centroid calibration
Use jobs/job_centroids_bis.sh

# Evaluate model perplexity/accuracy
Use jobs/job_test_model2.sh
