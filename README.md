<p align="center">
  <img src="images/banner.png" alt="Banner" width="100%">
</p>

<h1 align="center">⚡ TemporalKV: A Sub-4-bit KV-Cache Compression Framework ⚡</h1>

<p align="center">
  <b>Lightweight. Efficient. Accurate.</b>  
  <br>
  A framework for compressing KV-Caches in Large Language Models (LLMs) with sub-4-bit precision.
</p>

---

## 📖 Overview

This repository contains the implementation of **TemporalKV**, a framework for **sub-4-bit KV-Cache compression**.  
It enables efficient inference for large language models while maintaining strong accuracy.

---

## 📂 Repository Structure

- **`write_centroids_bis.py`** → Performs centroid calibration using collected activations.  
- **`test_model2.py`** → Integrates the compression framework in the LLM pipeline; used for perplexity & accuracy evaluation.  
- **`cache_utils.py`** → Custom low-precision cache implementation with outlier removal.  
- **`run-fisher2.py`** → Collects activations, gradients, and statistics (quantiles, normalization constants).  
- **`datautils.py`** → Utility functions for datasets and loaders.  
- **`results.csv`** → Experimental results (accuracy & perplexity).  
- **`jobs/`** → Shell scripts to launch experiments.  
- **`experiment_notebooks/`** → Jupyter notebooks for preliminary experiments.  
- **`fast_pytorch_kmeans/`** → Modified GPU K-Means (forked from [fast_pytorch_kmeans](https://github.com/DeMoriarty/fast_pytorch_kmeans)).  
- **`mses/`** → Mean squared error results as `.npy` arrays.  

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/temporal-kv.git
cd temporal-kv

# Install dependencies
pip install -r requirements.txt

# Run centroid calibration
python write_centroids_bis.py

# Evaluate model compression
python test_model2.py
