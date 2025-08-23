<p align="center">
  <img src="images/banner.png" alt="Banner" width="100%">
</p>

# A sub-4-bit KV-Cache compression framework

This repository contains the source code for the KV-Cache compression framework TemporalKV :

run-fisher2.py : file to collect activations, gradients and statistics such as quantiles and normalization constants
write_centroids_bis.py : file to perform centroid calibration using the collected activations
test_model2.py : file integrating the compression framework in the LLM pipeline. It is used to perform perplexity and accuracy measurement experiments
datautils.py : utils file

