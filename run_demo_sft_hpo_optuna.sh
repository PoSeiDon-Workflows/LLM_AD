#!/usr/bin/env bash

#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4144

module load conda
conda activate hf
export HF_EVALUATE_OFFLINE=1

cd /global/homes/p/papajim/GitHub/poseidon/LLM_AD
python3 /global/homes/p/papajim/GitHub/poseidon/LLM_AD/demo_sft_hpo_optuna.py

exit
