import glob
import shutil 
from pathlib import Path

# path = '/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim/VoxelPermutation'
path = '/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim/VoxelPermutation'
files = glob.glob(f'{path}/* *')
for src in files:
    dst = src.replace(' ', '_')
    shutil.move(src, dst)


files = glob.glob(f'{path}/dist/* *')
for src in files:
    dst = src.replace(' ', '_')
    shutil.move(src, dst)