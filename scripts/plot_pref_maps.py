import os
import nibabel as nib
import numpy as np
from nilearn import plotting, surface
from pathlib import Path
import argparse
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class PrefMaps:
    def __init__(self, args):
        self.process = 'PrefMaps'
        self.sid = str(args.s_num).zfill(2)
        self.cross_validation = args.CV
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        # Path(f'{args.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.models = ['indoor', 'expanse', 'transitivity', 'agent_distance', 'facingness', 'joint_action',
                       'communication', 'valence', 'arousal']
        myColors = ((0.8, 0.8, 0.8, 1.0), # gray - filler
                    (0.57421875, 0.51796875, 0.15, 1.0), # mustard - visual
                    (0.57421875, 0.51796875, 0.15, 1.0), # mustard - visual
                    (0.57421875, 0.51796875, 0.15, 1.0), # mustard - visual
                    (0.31171875, 0.20625, 0.571875, 1.0), #purle - social primitives
                    (0.31171875, 0.20625, 0.571875, 1.0), #purle - social primitives
                    (0.44921875, 0.8203125, 0.87109375, 1.0),  # cyan - social
                    (0.44921875, 0.8203125, 0.87109375, 1.0),  # cyan - social
                    (0.8515625, 0.32421875, 0.35546875, 1.0), # red - affective
                    (0.8515625, 0.32421875, 0.35546875, 1.0)) # red - affective
        self.cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(self.models)+1)

    def compute_surf_stats(self, hemi):
        stats = None
        for model in self.models:
            file_name = f'{self.out_dir}/SurfaceStats/sub-{self.sid}_prediction-all_drop-None_single-{model}_method-{self.method}_r2filtered_hemi-{hemi}.mgz'
            stat = surface.load_surf_data(file_name)
            if stats is None:
                stats = [np.zeros_like(stat)]
            stats.append(stat)
        stats = np.vstack(stats).argmax(axis=0)
        return stats

    def load_surf_mesh(self, hemi):
        return f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.inflated',\
               f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.sulc'

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi):

        view = plotting.view_surf(surf_mesh=surf_mesh,
                                  surf_map=surf_map,
                                  bg_map=bg_map,
                                  threshold=1,
                                  cmap=self.cmap,
                                  symmetric_cmap=False,
                                  title=f'sub-{self.sid}')
        view.save_as_html(f'{self.figure_dir}/sub-{self.sid}_method-{self.method}_hemi-{hemi}.html')

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
    parser.add_argument('--CV', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PrefMaps(args).run()

if __name__ == '__main__':
    main()
