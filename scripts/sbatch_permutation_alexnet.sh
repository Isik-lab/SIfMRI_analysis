#!/bin/bash -l

#SBATCH
#SBATCH --job-name=fmriprep
#SBATCH --time=25:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1
layer=$2

ml anaconda
conda activate nibabel

python voxel_permutation_alexnet.py -s "$subj" \
  --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
  --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
  --layer $layer

for roi in EVC MT EBA PPA FFA LOC pSTS face-pSTS aSTS TPJ; do
  time python roi_prediction_alexnet.py -s $subj \
    --roi $roi \
    --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
    --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
    --layer $layer
done

