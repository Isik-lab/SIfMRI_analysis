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


def model2vmax(model):
    d = {'arousal': 0.15,
         'communication': 0.1,
         'agent_distance': 0.075,
         'transitivity': 0.075}
    return d[model]


def roi_cmap():
    d = {
        'MT': (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
        'EBA': (1.0, 1.0, 0.0),
        'face-pSTS': (0.0, 0.8431372549019608, 1.0),
        'SI-pSTS': (1.0, 0.7686274509803922, 0.0)}
    return list(d.values())


class SurfaceStats:
    def __init__(self, args):
        self.process = 'SurfaceStats'
        self.sid = str(args.s_num).zfill(2)
        self.unique_model = args.unique_model
        self.single_model = args.single_model
        self.cross_validation = args.CV
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.ROIs = args.ROIs
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        if self.single_model is not None:
            self.figure_dir = f'{args.figure_dir}/{self.process}/single'
        elif self.unique_model is not None:
            self.figure_dir = f'{args.figure_dir}/{self.process}/variance_partitioning'
        else:
            self.figure_dir = f'{args.figure_dir}/{self.process}/full_model'
        self.file_name = f'sub-{self.sid}_prediction-all_drop-{self.unique_model}_single-{self.single_model}_method-{self.method}_r2filtered'
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        Path(f'{args.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.cmap = sns.color_palette('magma', as_cmap=True)
        self.rois = ['MT', 'EBA', 'face-pSTS', 'SI-pSTS']
        self.roi_cmap = roi_cmap()

    def compute_surf_stats(self, hemi):
        cmd = '/Applications/freesurfer/bin/mri_vol2surf '
        cmd += f'--src {self.out_dir}/VoxelPermutation/{self.file_name}.nii.gz '
        cmd += f'--out {self.out_dir}/{self.process}/{self.file_name}_hemi-{hemi}.mgz '
        cmd += f'--regheader sub-{self.sid} '
        cmd += f'--hemi {hemi} '
        cmd += '--projfrac 1'
        os.system(cmd)
        return surface.load_surf_data(f'{self.out_dir}/{self.process}/{self.file_name}_hemi-{hemi}.mgz')

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

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi):
        if hemi == 'lh':
            hemi_name = 'left'
        else:
            hemi_name = 'right'

        if self.ROIs:
            roi_map, roi_indices = self.load_rois(hemi)
            fig, ax = plt.subplots(1, figsize=(50, 50),
                                   subplot_kw={'projection': '3d'})
            plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                   roi_map=surf_map,
                                   bg_map=bg_map,
                                   vmax=model2vmax(self.unique_model),
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
                                        output_file=f'{self.figure_dir}/{self.file_name}_hemi-{hemi}.jpg')
        else:
            _, ax = plt.subplots(1, figsize=(50, 50),
                                 subplot_kw={'projection': '3d'})
            plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                   roi_map=surf_map,
                                   bg_map=bg_map,
                                   vmax=0.4,
                                   vmin=0.,
                                   axes=ax,
                                   cmap=self.cmap,
                                   hemi=hemi_name,
                                   view='lateral',
                                   output_file=f'{self.figure_dir}/{self.file_name}_hemi-{hemi}.jpg')
        print(f'{self.figure_dir}/{self.file_name}_hemi-{hemi}.jpg')

    def plot_one_hemi(self, hemi):
        surface_data = self.compute_surf_stats(hemi)
        inflated, sulcus = self.load_surf_mesh(hemi)
        self.plot_stats(inflated, sulcus, surface_data, hemi)

    def run(self):
        for hemi in ['lh', 'rh']:
            self.plot_one_hemi(hemi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--unique_model', type=str, default=None)
    parser.add_argument('--single_model', type=str, default=None)
    parser.add_argument('--CV', action=argparse.BooleanOptionalAction, default=False)
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
