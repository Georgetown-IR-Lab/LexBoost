# LexBoost

This repository contains the code to reproduce results in the paper "LexBoost: Improving Lexical Document Retrieval with Nearest Neighbors" accepted at The 24th ACM Symposium on Document Engineering (DocEng 2024).

In this work we extensively evaluate the proposed method LexBoost and show robustness and statistically significant improvements across baselines, datasets and corpus graph construction methods with virtually no additional latency overheads. We also show that LexBoost improves reranking results as well.

### lexboost-main.py
This code evaluates the proposed method on top of BM25 baseline at different fusion parameters, varying number of nearest neighbors from the TCT-ColBERT-HNP based corpus graph on the MSMARCO TREC DL 19 and 20 datasets.

### lexboost-baselines.py
This code evaluates the proposed method on top of BM25, PL2, DPH and QLD baselines at different fusion parameters, considering 16 nearest neighbors from the TCT-ColBERT-HNP based corpus graph on the MSMARCO TREC DL 19 and 20 datasets.

### lexboost-covid.py
This code evaluates the proposed method on top of BM25 baseline at different fusion parameters, varying number of nearest neighbors from the TCT-ColBERT-HNP based corpus graph on the CORD 19 - TREC COVID dataset.

### lexboost-alternategraph.py
This code evaluates the proposed method on top of BM25 baseline at different fusion parameters, varying number of nearest neighbors from the TCT-ColBERT-HNP based and TAS-B based corpus graphs on the MSMARCO TREC DL 19 and 20 datasets.

### lexboost-reranking.py
This code evaluates reranking using TAS-B on the proposed method on top of BM25 baseline at different fusion parameters, varying number of nearest neighbors from the TCT-ColBERT-HNP based corpus graph on the MSMARCO TREC DL 19 and 20 datasets.
