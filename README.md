# Causal Audio–Visual Embedding Learning

## Overview  
This repository contains the implementation of our causality-driven framework for learning robust audio–visual embeddings. We combine (1) Audio–Visual Semantic Alignment Loss (AV-SAL), (2) causal structure discovery via GRaSP, and (3) Causal Consistency Regularization (CCR) to disentangle true semantic dependencies from incidental co-occurrences.

## Features  
- Teacher–student training schedule with configurable transition epoch  
- AV-SAL for cross-modal soft-label alignment  
- GRaSP algorithm for directed causal graph inference  
- CCR loss for embedding regularization based on discovered causal links  
- Proxy-based triplet loss integration

## Requirements  
- Python 3.10+  
- PyTorch 1.12+  
- NumPy, SciPy, scikit-learn  
- GRaSP dependency (see `requirements.txt`)  
- CUDA-enabled GPU recommended

## Installation  
```bash
git clone https://github.com/me/causal-audio-visual-embedding.git
cd causal-audio-visual-embedding
pip install -r requirements.txt

More information will be upload soon.
