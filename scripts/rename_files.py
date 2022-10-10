import glob
import shutil 
from pathlib import Path

# path = '/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim/VoxelPermutation'
path = '/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_analysis/data/interim/VoxelPermutation'
files = glob.glob(f'{path}/* *')
for src in files:
    dst = src.replace(' ', '_')
    shutil.move(src, dst)

dist_dir = f'{path}/dist'
Path(dist_dir).mkdir(exist_ok=True, parents=True)
files = glob.glob(f'{path}/*r2var*')
for src in files:
    shutil.move(src, dist_dir)

files = glob.glob(f'{path}/*r2null*')
for src in files:
    shutil.move(src, dist_dir)

