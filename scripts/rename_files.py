import glob
import shutil

path = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/VoxelPermutation'
files = glob.glob(f'{path}/* *')
for src in files:
    dst = src.replace(' ', '_')
    shutil.move(src, dst)
