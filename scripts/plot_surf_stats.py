import os
import nibabel as nib
from nilearn import plotting, surface
from pathlib import Path
import argparse
import seaborn as sns

class SurfaceStats:
    def __init__(self, args):
        self.process = 'SurfaceStats'
        self.sid = str(args.s_num).zfill(2)
        self.unique_model = args.unique_model
        self.single_model = args.single_model
        if self.unique_model is not None:
            self.unique_model = self.unique_model.replace('_', ' ')
        if self.single_model is not None:
            self.single_model = self.single_model.replace('_', ' ')
        self.cross_validation = args.CV
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.file_name = f'sub-{self.sid}_prediction-all_drop-{self.unique_model}_single-{self.single_model}_method-{self.method}_r2'
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        Path(f'{args.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.cmap = sns.color_palette('magma', as_cmap=True)

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
        return f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.inflated',\
               f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.sulc'

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi):
        view = plotting.view_surf(surf_mesh=surf_mesh,
                                  surf_map=surf_map,
                                  bg_map=bg_map,
                                  threshold=1e-6,
                                  cmap=self.cmap,
                                  symmetric_cmap=False)
        view.save_as_html(f'{self.figure_dir}/{self.file_name}_hemi-{hemi}.html')

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
