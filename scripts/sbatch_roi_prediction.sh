#!/bin/bash -l

#SBATCH
#SBATCH --job-name=fmriprep
#SBATCH --time=10:0
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1
roi=$2
hemi=$3

ml anaconda
conda activate nibabel

for model in indoor expanse transitivity agent_distance facingness joint_action communication valence arousal; do
  python roi_prediction.py -s $subj --CV \
    --hemi $hemi --roi $roi --model $model \
    --out_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim \
    --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/raw \
    --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/reports/figures
done