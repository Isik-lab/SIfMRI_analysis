#!/bin/bash -l
#SBATCH --job-name=regress
#SBATCH --time=12:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=end

file=$1
feature="$2"

ml anaconda
conda activate nibabel

if [ -z ${feature+x} ]; then
  python voxel_permutation.py --y_pred "$file" \
    --out_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/interim \
    --data_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/raw
else
  python voxel_permutation.py --y_pred "$file" --pred_feature $feature \
    --out_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/interim \
    --data_dir /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/raw
fi