#!/bin/bash -l
#SBATCH
#SBATCH --job-name=regress
#SBATCH --time=5:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mail-type=end

sid=$1
num=$2

ml anaconda
conda activate nilearn
python voxelwise_encoding.py -s $sid -f $num \
-data /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SI_fMRI/input_data \
-output /home-2/emcmaho7@jhu.edu/work/mcmahoneg/SI_fMRI/output_data
