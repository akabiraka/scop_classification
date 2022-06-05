#!/usr/bin/sh

## this must be run from directory where run.py exists.
## --workdir is not used in this file.

#SBATCH --job-name=scop
#SBATCH --output=/scratch/akabir4/scop_classification/outputs/argo_logs/scop-%j.out
#SBATCH --error=/scratch/akabir4/scop_classification/outputs/argo_logs/scop-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

##cpu jobs
##SBATCH --partition=all-HiPri
##SBATCH --cpus-per-task=4
##SBATCH --mem=16000MB

##python files for CPU jobs
##python analyzers/plot_aa_embeddings.py

##GPU jobs
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --mem=32000MB

##nvidia-smi
##python files for GPU jobs
##python models/train_test.py
##python models/eval.py
##python models/save_model_outputs.py
python models/save_20_aa_embeddings.py