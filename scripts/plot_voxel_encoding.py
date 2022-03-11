#!/usr/bin/env python
# coding: utf-8

import argparse
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nilearn import datasets, surface
from nilearn.plotting import plot_surf_stat_map

from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
import itertools

from statsmodels.stats.multitest import multipletests

def correct(ps_, rs_, p_crit=1e-2):
    sig_bool, ps_corrected, _, _ = multipletests(ps_, alpha=p_crit, method='fdr_by')
    indices = np.where(sig_bool)[0]
    return sig_bool, rs_[indices].min()


def filter_r(rs, ps):
    ps, threshold = correct(ps, rs)
    ps = np.invert(ps)
    indices = np.where(ps)[0]
    rs[indices] = 0
    return rs, threshold


def save(arr, out_name, mode='npy'):
    if mode == 'npy':
        np.save(out_name, arr)
    elif mode == 'nii':
        nib.save(arr, out_name)


class PlotEncoding():
    def __init__(self, args):
        self.process = 'PlotEncoding'
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.feature = args.feature
        self.cmap = sns.color_palette(args.seaborn_palette, as_cmap=True)
        self.stat_dir = args.stat_dir
        self.mask_dir = args.mask_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)

    def mkNifti(self, mask, arr, im, zeros=True):
        if zeros:
            out_im = np.zeros(mask.size)
        else:
            out_im = np.ones(mask.size)
        inds = np.where(mask)[0]
        out_im[inds] = arr
        return nib.Nifti1Image(out_im.reshape(im.shape), affine=im.affine)

    def _colorbar_from_array(self, array, threshold, cmap):
        """Generate a custom colorbar for an array.
        Internal function used by plot_img_on_surf
        array : np.ndarray
            Any 3D array.
        vmax : float
            upper bound for plotting of stat_map values.
        threshold : float
            If None is given, the colorbar is not thresholded.
            If a number is given, it is used to threshold the colorbar.
            Absolute values lower than threshold are shown in gray.
        kwargs : dict
            Extra arguments passed to _get_colorbar_and_data_ranges.
        cmap : str, optional
            The name of a matplotlib or nilearn colormap.
            Default='cold_hot'.
        """
        vmin = array.min()
        vmax = array.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmaplist = [cmap(i) for i in range(cmap.N)]

        if threshold is None:
            threshold = 0.

        # set colors to grey for absolute values < threshold
        istart = int(vmin)
        istop = int(norm(threshold, clip=True) * (cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)
        our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                     cmaplist, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=our_cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # fake up the array of the scalar mappable.
        sm._A = []

        return sm

    def plot_stats(self, texture,
                   title=None,
                   modes=['lateral', 'medial', 'ventral'],
                   hemis=['left', 'right'],
                   cmap=None, threshold=0.01,
                   output_file=None, colorbar=True,
                   vmax=None, kwargs={}):

        cbar_h = .25
        title_h = .25 * (title is not None)
        # Set the aspect ratio, but then make the figure twice as big to increase resolution
        w, h = plt.figaspect((len(modes) + cbar_h + title_h) / len(hemis)) * 2
        fig = plt.figure(figsize=(w, h), constrained_layout=False)
        height_ratios = [title_h] + [1.] * len(modes) + [cbar_h]
        grid = gridspec.GridSpec(
            len(modes) + 2, len(hemis),
            left=0., right=1., bottom=0., top=1.,
            height_ratios=height_ratios, hspace=0.0, wspace=0.0)
        axes = []
        for i, (mode, hemi) in tqdm(enumerate(itertools.product(modes, hemis)),
                                    total=len(modes)*len(hemis)):
            bg_map = self.fsaverage['sulc_%s' % hemi]
            ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
            axes.append(ax)
            plot_surf_stat_map(self.fsaverage[f'infl_{hemi}'],
                               texture[hemi],
                               view=mode, hemi=hemi,
                               bg_map=bg_map,
                               axes=ax,
                               colorbar=False,  # Colorbar created externally.
                               vmax=vmax,
                               threshold=threshold,
                               cmap=cmap,
                               **kwargs)
            # We increase this value to better position the camera of the
            # 3D projection plot. The default value makes meshes look too small.
            ax.dist = 7

        if colorbar:
            array = np.hstack((texture['left'], texture['right']))
            sm = self._colorbar_from_array(array, threshold, get_cmap(cmap))

            cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
            cbar_ax = fig.add_subplot(cbar_grid[1])
            axes.append(cbar_ax)
            fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

        if title is not None:
            fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

        if output_file is not None:
            fig.savefig(output_file, bbox_inches="tight")
            plt.close(fig)

    def vol_to_surf(self, im, hemi):
        return surface.vol_to_surf(im, surf_mesh=self.fsaverage[f'pial_{hemi}'], radius=2.)

    def load(self):
        mask_im = nib.load(f'{self.mask_dir}/sub-all_stat-rho_statmap.nii.gz')
        mask = np.load(f'{self.mask_dir}/sub-all_reliability-mask.npy')
        rs = np.load(f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-{self.feature}_rs.npy')
        ps = np.load(f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-{self.feature}_ps.npy')
        rs, threshold = filter_r(rs, ps)
        save(rs, f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-{self.feature}_rs-filtered.npy')
        rs = self.mkNifti(mask, rs, mask_im)
        save(rs, f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-{self.feature}_rs-filtered.nii.gz',
             mode='nii')

        texture = {'left': self.vol_to_surf(rs, 'left'),
                   'right': self.vol_to_surf(rs, 'right')}
        return texture, threshold

    def run(self):
        # load reliability files
        texture, threshold = self.load()
        name = f'{self.figure_dir}/sub-{self.sid}.png'
        self.plot_stats(texture, cmap=self.cmap,
                        output_file=name, threshold=threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--feature', '-f', type=str, default='all')
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--seaborn_palette', '-palette', type=str, default='magma')
    parser.add_argument('--mask_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/Reliability')
    parser.add_argument('--stat_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/VoxelPermutation')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PlotEncoding(args).run()

if __name__ == '__main__':
    main()
