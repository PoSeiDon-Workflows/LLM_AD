#!/usr/bin/bash

# >>> load module on swing (cuda 11.8)
# module load cuda
# <<< 

conda create -n hf python=3.11 -y
source activate hf

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
      pyg transformers evaluate \
      autopep8 flake8 ipykernel ipywidgets umap-learn pytest \
      -c pytorch -c nvidia -c huggingface -c pyg -y

pip install accelerate adjustText optuna deephyper -U

# NERSC specific
# module load conda

# conda create -n hf python=3.11 -y
# source activate hf

# conda install pytorch=2.1.0 pyg -c pytorch -c nvidia -c pyg -y

# pip install transformers evaluate datasets autopep8 flake8 ipykernel ipywidgets umap-learn pytest accelerate adjustText optuna deephyper -U