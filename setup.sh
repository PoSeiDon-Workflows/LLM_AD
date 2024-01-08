#!/usr/bin/bash

conda create -n hf python=3.11 -y
source activate hf

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 \
      pyg transformers evaluate \
      autopep8 flake8 ipykernel ipywidgets umap-learn pytest \
      -c pytorch -c nvidia -c huggingface -c pyg -y

pip install accelerate adjustText -U