#!/bin/bash -l

#SBATCH
#SBATCH --job-name=fmriprep
#SBATCH --time=50:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1
feature=$2

ml anaconda
conda activate nibabel

# s

for roi in EVC MT EBA PPA FFA LOC pSTS face-pSTS aSTS; do
  time python roi_prediction.py -s $subj \
    --roi $roi \
    --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
    --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
    --feature $feature --unique_variance --include_nuisance
done