import glob
import os
import numpy as np
from nilearn import plotting, surface, image
from pathlib import Path
import argparse
import seaborn as sns
import nibabel as nib
from scipy import ndimage
from src.tools import camera_switcher


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


def get_vmax(analysis):
    d = dict(full=0.7,
             categories=0.6,
             features=0.25,
             categories_unique=0.25,
             features_unique=0.15)
    return d[analysis]


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
        self.vmax = 1
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
        self.base = ''

    def get_file_names(self):
        if self.unique_variance:
            if self.category is not None:
                base = f'sub-{self.sid}_dropped-categorywithnuisance-{self.category}'
                analysis = 'categories_unique'
            else:  # self.feature is not None:
                base = f'sub-{self.sid}_dropped-featurewithnuisance-{self.feature}'
                analysis = 'features_unique'
        else:  # not self.unique_variance
            if self.category is not None:
                # Regression with the categories without the other regressors
                base = f'sub-{self.sid}_category-{self.category}'
                analysis = 'categories'
            elif self.feature is not None:
                base = f'sub-{self.sid}_feature-{self.feature}'
                analysis = 'features'
            else:  # This is the full regression model with all annotated features
                base = f'sub-{self.sid}_full-model'
                analysis = 'full'
        self.vmax = get_vmax(analysis)
        self.in_file_prefix = f'{self.out_dir}/VoxelPermutation/{base}_r2filtered.nii.gz'
        self.out_file_prefix = f'{self.out_dir}/{self.process}/{base}'
        Path(f'{self.figure_dir}/{analysis}/sub-{self.sid}').mkdir(parents=True, exist_ok=True)
        self.figure_prefix = f'{self.figure_dir}/{analysis}/sub-{self.sid}/{base}'
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

        surf_map = np.nan_to_num(surf_map)
        surf_map[surf_map < 0] = 0
        if np.sum(np.invert(np.isclose(surf_map, 0))) > 0:
            threshold = surf_map[np.invert(np.isclose(surf_map, 0))].min()
        else:
            threshold = 0
        max_val = surf_map.max()
        print(f'smallest value = {threshold:.3f}')
        print(f'largest value = {max_val:.3f}')
        for view in ['ventral', 'lateral', 'medial']:
            colorbar = True if view == 'lateral' and hemi_ == 'rh' else False
            fig = plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                         roi_map=surf_map,
                                         bg_map=bg_map,
                                         vmax=self.vmax,
                                         threshold=threshold,
                                         engine='plotly',
                                         colorbar=colorbar,
                                         view=view,
                                         cmap=self.cmap,
                                         hemi=hemi_name)
            fig.figure.update_layout(scene_camera=camera_switcher(hemi_, view),
                                     paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)")
            fig.figure.write_image(f'{self.figure_prefix}_view-{view}_hemi-{hemi_}.png')

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
