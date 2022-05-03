#!/bin/bash -l
#SBATCH --job-name=permutation
#SBATCH --time=30:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=end

model=$1

ml anaconda
conda activate nibabel

python voxel_group_results.py -m "$model" --cross_validated \
  --out_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/interim \
  --data_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/raw