#!/bin/bash -l

#SBATCH
#SBATCH --job-name=fmriprep
#SBATCH --time=1:30:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1
model=$2

ml anaconda
conda activate nibabel

python voxel_permutation.py -s "$subj" \
  --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
  --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
  --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/reports/figures \
  --unique_model $model

for roi in EVC MT EBA face-pSTS SI-pSTS TPJ; do
for hemi in lh rh; do
  time python roi_prediction.py -s $subj \
    --hemi $hemi --roi $roi \
    --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
    --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
    --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/reports/figures \
    --model $model
done; done