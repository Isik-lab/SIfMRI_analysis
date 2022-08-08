SID=1
DROP=communication
SINGLE=None
METHOD=CV
HEMI=left
PATH=/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/VoxelPermutation

/Applications/freesurfer/bin/mri_vol2surf \
--src $PATH/sub-0${SID}_prediction-all_drop-${DROP}_single-${SINGLE}_method-${METHOD}_r2.nii.gz \
--out $PATH/sub-0${SID}_prediction-all_drop-${DROP}_single-${SINGLE}_method-${METHOD}_hemi-${HEMI}_r2.mgh \
--regheader sub-0${SID} \
--hemi lh
