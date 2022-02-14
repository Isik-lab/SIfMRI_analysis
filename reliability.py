import argparse
import os
import glob
import numpy as np
import pandas as pd

from nilearn import surface, datasets, plotting
import matplotlib.pyplot as plt
import nibabel as nib

def corr2d(a, b):
    a_m = a - a.mean(axis=0)
    b_m = b - b.mean(axis=0)

    r = np.zeros(b.shape[0])
    for i in range(b.shape[0]):
        r[i] = (a_m[i, :] @ b_m[i, :]) / (np.sqrt((a_m[i, :] @ a_m[i, :]) * (b_m[i, :] @ b_m[i, :])))
    return r

class reliability():
    def __init__(self, args):
        self.sid = sid = str(args.s_num).zfill(2)
        self.process = 'reliability'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}/sub-{self.sid}'
        self.figure_dir = f'{args.figure_dir}/{self.process}/sub-{self.sid}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def run(self, threshold=0.279):
        test_videos = pd.read_csv(f'{self.data_dir}/annotations/test.csv')
        files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*beta.npy'))

        # Find the test runs
        runs = []
        for f in files:
            if test_videos.iloc[0,0].split('.mp4')[0] in f:
                runs.append(f.split('run-')[-1].split('_')[0])

        # Save info about number of runs and conditions
        half = int(len(runs)/2)
        nconds = len(test_videos)

        # Load an ROI file to get meta data about the images
        im = nib.load(f'{self.data_dir}/ROI_masks/sub-{self.sid}/sub-{self.sid}_region-EVC_mask.nii.gz')
        vol = im.shape
        n_voxels = np.prod(vol)
        affine = im.affine

        even = np.zeros((n_voxels, nconds))
        odd = np.zeros((n_voxels, nconds))
        for ri, run in enumerate(runs):
            # Get all the files for the current run
            files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*run-{run}*beta.npy'))
            # Initialize an empty array for the current run
            arr = np.zeros((n_voxels, nconds))
            fi = 0
            # Append all conditions to the current array, except for the crowd condition
            for f in files:
                if not 'crowd' in f:
                    arr[..., fi] = np.load(f).flatten()
                    fi += 1

            # If the run is an even repeat, add to even array, else add to odd
            if ri % 2:
                even += arr
            else:
                odd += arr

        # Get the average for each of the voxels
        even /= half
        odd /= half

        # Compute the correlation
        r_map = corr2d(even, odd)

        # Make the array into a nifti image and save
        r_im = nib.Nifti1Image(np.array(r_map).reshape(vol), affine)
        r_name = f'{self.out_dir}/sub-{self.sid}_stat-rho_statmap.nii.gz'
        nib.save(r_im, r_name)

        #Save the numpy array
        r_name = f'{self.out_dir}/sub-{self.sid}_stat-rho_statmap.npy'
        np.save(r_name, r_im)

        #vol to surf
        fsaverage = datasets.fetch_surf_fsaverage()
        r_surf = surface.vol_to_surf(r_im, fsaverage.pial_right)

        # Plot in the volume
        plotting.plot_stat_map(r_im, display_mode='ortho',
                               threshold=threshold,
                               output_file=f'{self.figure_dir}/sub-{self.sid}_view-volume_stat-rho_statmap.pdf')

        # Plot on the surface
        cmap = plt.get_cmap('bwr')
        plotting.plot_surf_stat_map(fsaverage.infl_right, r_surf,
                                    hemi='right', bg_map=fsaverage.sulc_right,
                                    colorbar=True, cmap=cmap,
                                    threshold=threshold,
                                    output_file=f'{self.figure_dir}/sub-{self.sid}_hemi-rh_view-surflat_stat-rho_statmap.pdf')
        plotting.plot_surf_stat_map(fsaverage.infl_right, r_surf,
                                    hemi='right', bg_map=fsaverage.sulc_right,
                                    colorbar=True, cmap=cmap,
                                    threshold=threshold, view='ventral',
                                    output_file=f'{self.figure_dir}/sub-{self.sid}_hemi-rh_view-surfvent_stat-rho_statmap.pdf')
        plotting.plot_surf_stat_map(fsaverage.infl_left, r_surf,
                                    hemi='left', bg_map=fsaverage.sulc_left,
                                    colorbar=True, cmap=cmap,
                                    threshold=threshold,
                                    output_file=f'{self.figure_dir}/sub-{self.sid}_hemi-lh_view-surflat_stat-rho_statmap.pdf')
        plotting.plot_surf_stat_map(fsaverage.infl_left, r_surf,
                                    hemi='left', bg_map=fsaverage.sulc_left,
                                    colorbar=True, cmap=cmap,
                                    threshold=threshold, view='ventral',
                                    output_file=f'{self.figure_dir}/sub-{self.sid}_hemi-lh_view-surfvent_stat-rho_statmap.pdf')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    parser.add_argument('--figure_dir', '-figures', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/figures')
    args = parser.parse_args()
    reliability(args).run()

if __name__ == '__main__':
    main()
