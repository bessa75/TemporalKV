<p align="center">
  <img src="images/banner.png" alt="Banner" width="100%">
</p>

# A sub-4-bit KV-Cache compression framework

This repository contains the source code for the KV-Cache compression framework TemporalKV :

write_centroids_bis.py : file to perform centroid calibration using the collected activations

test_model2.py : file integrating the compression framework in the LLM pipeline. It is used to perform perplexity and accuracy measurement experiments

cache_utils.py : file containing our custom low-precision Cache implementation, along with the outlier removal framework

run-fisher2.py : file to collect activations, gradients and statistics such as quantiles and normalization constants

datautils.py : utils file

results.csv : results from our experiments

jobs : .sh files to run experiments

experiment_notebooks : notebooks used to run preliminary experiments

fast_pytorch_kmeans : modified GPU K-Means implementation (forked from https://github.com/DeMoriarty/fast_pytorch_kmeans)

mses : mean mse results as .npy arrays
