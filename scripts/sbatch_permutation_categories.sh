#!/bin/bash -l

#SBATCH
#SBATCH --job-name=fmriprep
#SBATCH --time=5:0
#SBATCH --partition=defq
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1
category=$2

ml anaconda
conda activate nibabel

#python voxel_permutation_categories.py -s "$subj" \
#  --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
#  --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
#  --category $category

for roi in PPA EVC MT EBA face-pSTS SI-pSTS TPJ; do
for hemi in lh rh; do
  time python roi_category.py -s $subj \
    --hemi $hemi --roi $roi \
    --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
    --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
    --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/reports/figures \
    --category $category
done; done


