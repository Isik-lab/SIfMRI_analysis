import glob
import os
import numpy as np
from nilearn import plotting, surface
from pathlib import Path
import argparse
import seaborn as sns
import nibabel as nib
from src.tools import camera_switcher
from src.custom_plotting import custom_nilearn_cmap
import matplotlib as mpl


class PrefMap:
    def __init__(self, args):
        self.process = 'PrefMap'
        self.sid = str(args.s_num).zfill(2)
        self.overwrite = args.overwrite
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.unique_variance = args.unique_variance
        if self.unique_variance:
            self.figure_prefix = f'{self.figure_dir}/sub-{self.sid}_uniquecategory_preference'
            self.volume_map = f'{self.out_dir}/{self.process}/sub-{self.sid}_uniquecategory-preference-map.nii.gz'
            self.surf_map = f'{self.out_dir}/{self.process}/sub-{self.sid}_uniquecategory-preference-map_hemi'
            self.infile_prefix = f'{self.out_dir}/VoxelPermutation/sub-{self.sid}_dropped-categorywithnuisance'
        else:
            self.figure_prefix = f'{self.figure_dir}/sub-{self.sid}_category_preference'
            self.volume_map = f'{self.out_dir}/{self.process}/sub-{self.sid}_category-preference-map.nii.gz'
            self.surf_map = f'{self.out_dir}/{self.process}/sub-{self.sid}_category-preference-map_hemi'
            self.infile_prefix = f'{self.out_dir}/VoxelPermutation/sub-{self.sid}_category'
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        Path(f'{args.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        print(vars(self))
        self.cmap = custom_nilearn_cmap()
        # self.cmap = mpl.colormaps['Accent']
        self.categories = ['alexnet', 'moten', 'scene_object', 'social_primitive', 'social', 'affective']

    def compute_preference_map(self):
        if not os.path.exists(self.volume_map) or self.overwrite:
            map = None
            for icat, category in enumerate(self.categories):
                file = f'{self.infile_prefix}-{category}_r2filtered.nii.gz'
                brain = nib.load(file)
                brain_arr = np.array(brain.dataobj)
                if map is None:
                    map = np.zeros((brain_arr.size, len(self.categories)))
                map[:, icat] = brain_arr.flatten()
            nonzero_inds = np.invert(np.isclose(np.sum(map, axis=-1), 0))
            out_map = np.zeros(brain_arr.size)
            out_map[nonzero_inds] = np.argmax(map[nonzero_inds, :], axis=-1) + 1
            out_map = nib.Nifti1Image(out_map.reshape(brain.shape),
                                      affine=brain.affine, header=brain.header)
            nib.save(out_map, self.volume_map)

    def compute_surf_stats(self, hemi_):
        surface_file = f'{self.surf_map}-{hemi_}.mgz'
        if not os.path.exists(surface_file) or self.overwrite:
            cmd = '/Applications/freesurfer/bin/mri_vol2surf '
            cmd += f'--src {self.volume_map} '
            cmd += f'--out {surface_file} '
            cmd += f'--regheader sub-{self.sid} '
            cmd += f'--hemi {hemi_} '
            cmd += '--projfrac 1'
            os.system(cmd)
        return surface.load_surf_data(surface_file)

    def load_surf_mesh(self, hemi):
        return f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.inflated', \
               f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.sulc'

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi_):
        if hemi_ == 'lh':
            hemi_name = 'left'
        else:
            hemi_name = 'right'

        fig = plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                     roi_map=surf_map,
                                     bg_map=bg_map,
                                     engine='plotly',
                                     threshold=1.,
                                     colorbar=True,
                                     cmap=self.cmap,
                                     hemi=hemi_name)
        fig.figure.write_html(f'{self.figure_prefix}_hemi-{hemi_}.html')

        for view in ['lateral', 'ventral', 'medial']:
            colorbar = True if view == 'lateral' and hemi_ == 'rh' else False
            fig = plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                         roi_map=surf_map,
                                         bg_map=bg_map,
                                         engine='plotly',
                                         threshold=1.,
                                         colorbar=colorbar,
                                         view=view,
                                         cmap=self.cmap,
                                         hemi=hemi_name)
            fig.figure.update_layout(scene_camera=camera_switcher(hemi_, view),
                                     paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)")
            fig.figure.write_image(f'{self.figure_prefix}_view-{view}_hemi-{hemi_}.png')

    def run(self):
        self.compute_preference_map()
        for hemi in ['rh', 'lh']:
            surface_data = self.compute_surf_stats(hemi)
            inflated, sulcus = self.load_surf_mesh(hemi)
            self.plot_stats(inflated, sulcus, surface_data, hemi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default=2)
    parser.add_argument('--unique_variance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PrefMap(args).run()


if __name__ == '__main__':
    main()
