#!/bin/bash
#SBATCH --job-name=scopH
#SBATCH --output=/scratch/akabir4/scop_classification/outputs/argo_logs/scopH-%j.out
#SBATCH --error=/scratch/akabir4/scop_classification/outputs/argo_logs/scopH-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

## cpu 
#SBATCH --partition=normal                  # submit   to the normal(default) partition
##SBATCH --cpus-per-task=8                   # Request n   cores per node
##SBATCH --nodes=1                          # Request N nodes
#SBATCH --mem=16000MB                # Request nGB RAM per core
##SBATCH --array=0-40000                         # distributed array job   

## python generators/DownloadCleanFasta.py
python generators/Features.py

## gpu
##SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
##SBATCH --qos=gpu                           # need to select 'gpu' QoS
###SBATCH --nodes=1
##SBATCH --ntasks-per-node=1                 # up to 128; 
##SBATCH --gres=gpu:A100.40gb:1              # up to 8; only request what you need
##SBATCH --mem-per-cpu=32000MB               # memory per CORE; total memory is 1 TB (1,000,000 MB)

## cpu jobs
##python generators/DownloadCleanFasta.py
##python generators/data_helper.py
##python generators/Features.py

## gpu jobs
##nvidia-smi
##python models/train_test.py
##python models/eval.py
