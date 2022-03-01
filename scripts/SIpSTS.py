#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
from scipy import stats
import nibabel as nib
from nilearn import plotting

class SIpSTS():
    def __init__(self, args):
        self.process = 'SIpSTS'
        self.data_dir = args.data_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def run(self, p=5e-2):
        for sid_ in range(4):
            sid = str(sid_ + 1).zfill(2)
            stats_file = f'{self.data_dir}/SIpSTS/stats.sub-{sid}.SIpSTS_REML+tlrc.HEAD'
            brik = nib.load(stats_file)
            labels = brik.header.get_volume_labels()

            t_map = labels.index('Interact-Non#0_Tstat')
            t_map = nib.Nifti1Image(brik.dataobj[..., t_map], affine=brik.affine)
            # coef = labels.index('Interact-Non#0_Coef')
            # coef = nib.Nifti1Image(brik.dataobj[..., coef], affine=brik.affine)

            threshold = np.abs(stats.t.ppf(p, 15))
            out_name = f'{self.figure_dir}/sub-{sid}.png'
            plotting.plot_img_on_surf(t_map, inflate=True,
                                      surf_mesh='fsaverage',
                                      threshold=threshold,
                                      output_file=out_name,
                                      views=['lateral', 'medial', 'ventral'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    SIpSTS(args).run()

if __name__ == '__main__':
    main()
