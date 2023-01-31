import glob
import os
import numpy as np
from nilearn import plotting, surface, image
from pathlib import Path
import argparse
import seaborn as sns
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt


def roi2contrast(roi):
    d = dict()
    d['MT'] = 'motionVsStatic'
    d['face-pSTS'] = 'facesVsObjects'
    d['EBA'] = 'bodiesVsObjecs'
    d['PPA'] = 'scenesVsObjects'
    d['TPJ'] = 'beliefVsPhoto'
    d['SI-pSTS'] = 'interactVsNoninteract'
    d['EVC'] = 'EVC'
    return d[roi]


def roi_cmap():
    d = {
        'MT': (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
        'EBA': (1.0, 1.0, 0.0),
        'LOC': (0., 1.0, 1.0),
        'face-pSTS': (0.0, 0.8431372549019608, 1.0),
        'pSTS': (1.0, 1.0, 1.0),
        'aSTS': (1.0, 1.0, 0.)}
    return list(d.values())


class SurfaceStats:
    def __init__(self, args):
        self.process = 'SurfaceStats'
        self.sid = str(args.s_num).zfill(2)
        self.unique_variance = args.unique_variance
        self.feature = args.feature
        self.category = args.category
        self.ROIs = args.ROIs
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        Path(f'{args.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        print(vars(self))
        self.cmap = sns.color_palette('magma', as_cmap=True)
        self.rois = ['MT', 'EBA', 'LOC', 'face-pSTS', 'pSTS', 'aSTS']
        self.roi_cmap = roi_cmap()
        self.in_file_prefix = ''
        self.out_file_prefix = ''
        self.figure_prefix = ''

    def get_file_names(self):
        if self.unique_variance:
            if self.category is not None:
                base = f'sub-{self.sid}_dropped-categorywithnuisance-{self.category}'
            else:  # self.feature is not None:
                base = f'sub-{self.sid}_dropped-featurewithnuisance-{self.feature}'
        else:  # not self.unique_variance
            if self.category is not None:
                # Regression with the categories without the other regressors
                base = f'sub-{self.sid}_category-{self.category}'
            elif self.feature is not None:
                base = f'sub-{self.sid}_feature-{self.feature}'
            else:  # This is the full regression model with all annotated features
                base = f'sub-{self.sid}_full-model'
        self.in_file_prefix = f'{self.out_dir}/VoxelPermutation/{base}_r2filtered.nii.gz'
        self.out_file_prefix = f'{self.out_dir}/{self.process}/{base}'
        self.figure_prefix = f'{self.figure_dir}/{base}'
        print(self.in_file_prefix)
        print(self.out_file_prefix)
        print(self.figure_prefix)

    def compute_surf_stats(self, hemi_):
        file = f'{self.out_file_prefix}_hemi-{hemi_}.mgz'
        if not os.path.exists(file):
            cmd = '/Applications/freesurfer/bin/mri_vol2surf '
            cmd += f'--src {self.in_file_prefix} '
            cmd += f'--out {file} '
            cmd += f'--regheader sub-{self.sid} '
            cmd += f'--hemi {hemi_} '
            cmd += '--projfrac 1'
            os.system(cmd)
        return surface.load_surf_data(file)

    def load_surf_mesh(self, hemi):
        return f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.inflated', \
               f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.sulc'

    def load_rois(self, hemi):
        vol_out_file = f'{self.out_dir}/Localizers/sub-{self.sid}_combined-roi_mask-vol_hemi-{hemi}.nii.gz'
        surf_out_file = f'{self.out_dir}/Localizers/sub-{self.sid}_combined-roi_mask-surf_hemi-{hemi}.mgz'
        print(vol_out_file + '\n \n')
        if not os.path.exists(vol_out_file):
            combined_mask = None
            for i_roi, roi in enumerate(self.rois):
                contrast = roi2contrast(roi)
                file = glob.glob(f'{self.data_dir}/localizers/sub-{self.sid}/*{contrast}*{hemi}*.nii.gz')[0]
                img = nib.load(file)
                current_mask = np.array(img.dataobj).astype('bool')
                print(f'{hemi} {roi} size: {np.sum(current_mask)}')

                label, n = ndimage.label(current_mask)
                for i in range(1, n + 1):
                    if np.sum(label == i) < 10:
                        current_mask[label == i] = False
                current_mask = ndimage.binary_closing(current_mask, iterations=2)
                print(f'revised {hemi} {roi} size: {np.sum(current_mask)}')

                if combined_mask is None:
                    combined_mask = np.zeros(current_mask.shape, dtype='float')
                combined_mask[current_mask] = i_roi + 1

            combined_mask = image.new_img_like(img, combined_mask)
            nib.save(combined_mask, vol_out_file)

            cmd = '/Applications/freesurfer/bin/mri_vol2surf '
            cmd += f'--src {vol_out_file} '
            cmd += f'--out {surf_out_file} '
            cmd += f'--regheader sub-{self.sid} '
            cmd += f'--hemi {hemi} '
            cmd += '--projfrac 1'
            os.system(cmd)
        return surface.load_surf_data(surf_out_file), (np.arange(len(self.rois)) + 1).tolist()

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi_):
        if hemi_ == 'lh':
            hemi_name = 'left'
        else:
            hemi_name = 'right'

        if self.ROIs:
            roi_map, roi_indices = self.load_rois(hemi_)
            fig, ax = plt.subplots(1, figsize=(50, 50),
                                   subplot_kw={'projection': '3d'})
            plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                   roi_map=surf_map,
                                   bg_map=bg_map,
                                   vmax=0.5,
                                   vmin=0.,
                                   cmap=self.cmap,
                                   axes=ax,
                                   colorbar=False,
                                   hemi=hemi_name,
                                   view='lateral')
            plotting.plot_surf_contours(surf_mesh=surf_mesh,
                                        roi_map=roi_map,
                                        legend=False,
                                        labels=self.rois,
                                        levels=roi_indices,
                                        figure=fig,
                                        axes=None,
                                        colors=self.roi_cmap,
                                        output_file=f'{self.figure_prefix}_hemi-{hemi_}.jpg')
        else:
            for view in ['lateral', 'ventral', 'medial']:
                _, axes = plt.subplots(1, figsize=(10, 10),
                                     subplot_kw={'projection': '3d'})
                # for ax, view in zip(axes, views):
                plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                       roi_map=surf_map,
                                       bg_map=bg_map,
                                       vmax=0.5,
                                       vmin=0.,
                                       axes=axes,
                                       cmap=self.cmap,
                                       hemi=hemi_name,
                                       colorbar=False,
                                       view=view)
                plt.savefig(f'{self.figure_prefix}_view-{view}_hemi-{hemi_}.jpg')

    def plot_one_hemi(self, hemi_):
        surface_data = self.compute_surf_stats(hemi_)
        inflated, sulcus = self.load_surf_mesh(hemi_)
        self.plot_stats(inflated, sulcus, surface_data, hemi_)

    def run(self):
        self.get_file_names()
        for hemi in ['lh', 'rh']:
            self.plot_one_hemi(hemi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default=1)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--feature', type=str, default=None)
    parser.add_argument('--unique_variance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ROIs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    SurfaceStats(args).run()


if __name__ == '__main__':
    main()
