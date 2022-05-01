#!/bin/bash -l
#SBATCH --job-name=permutation
#SBATCH --time=1:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=end

subj=$1
model=$2

ml anaconda
conda activate nibabel

python voxel_permutation.py -s "$subj" -m "$model" \
  --out_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/interim \
  --data_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/raw