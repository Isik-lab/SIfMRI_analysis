import glob
import shutil

path = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/VoxelPermutation'
files = glob.glob(f'{path}/*.nii.gz')
print(files)
for file in files:
    if ' ' in file:
        print(file)
        new=file.replace(' ', '_')
        shutil.move(file, new)
