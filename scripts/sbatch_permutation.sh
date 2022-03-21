#!/bin/bash -l
#SBATCH --job-name=regress
#SBATCH --time=12:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=end

sid=$1
feature=$2
control=$3

ml anaconda
conda activate nibabel
if [ "$feature" = "not_by_feature" ]; then
  python voxel_permutation.py -s $sid \
  -data /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/raw \
  -output /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/interim \
  --control "$control"
else
  python voxel_permutation.py -s $sid \
  -data /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/raw \
  -output /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SIfMRI_analysis/data/interim \
  --by_feature \
  --feature "$feature" \
  --control "$control"
fi