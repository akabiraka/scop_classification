#!/bin/bash
#SBATCH --job-name=prot
#SBATCH --output=/scratch/akabir4/scop_classification/outputs/argo_logs/prot-%N-%j.out
#SBATCH --error=/scratch/akabir4/scop_classification/outputs/argo_logs/prot-%N-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

## cpu 
#SBATCH --partition=normal                  # submit   to the normal(default) partition
#SBATCH --cpus-per-task=8                   # Request n   cores per node
##SBATCH --nodes=1                          # Request N nodes
#SBATCH --mem-per-cpu=4000MB                # Request nGB RAM per core
#SBATCH --array=0-5                         # distributed array job   

## gpu
##SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
##SBATCH --qos=gpu                           # need to select 'gpu' QoS
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1                 # up to 128; 
##SBATCH --gres=gpu:A100.40gb:1              # up to 8; only request what you need
##SBATCH --mem-per-cpu=16000MB               # memory per CORE; total memory is 1 TB (1,000,000 MB)

##nvidia-smi
python generators/DownloadCleanFasta.py
##python generators/Features.py