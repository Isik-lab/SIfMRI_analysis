#!/usr/bin/env python
# coding: utf-8

import argparse
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface
from nilearn.plotting import plot_surf_roi

from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
import itertools
from statsmodels.stats.multitest import multipletests


def save(arr, out_name, mode='npy'):
    if mode == 'npy':
        np.save(out_name, arr)
    elif mode == 'nii':
        nib.save(arr, out_name)


def mk_cmap(n_features=12):
    # colors = ['blue', 'purple', 'yellow']
    # colors = ['#48D4E1', '#8C55FD', '#FADC00']
    c1 = tuple(np.array([72., 212., 225.]) / 256)
    c2 = tuple(np.array([140., 85., 253.]) / 256)
    c3 = tuple(np.array([250., 220., 0.]) / 256)

    cmap = sns.color_palette('Paired', n_features, as_cmap=True)
    colors = []
    for c in range(n_features):
        # if c <= 1 or c == 4:
        if c <= 4:
            colors.append(c1)
            cmap._lut[c] = list(c1) + [1.]
        elif c > 4 and c <= 6:
        # elif c == 2 or c == 3:
            colors.append(c2)
            cmap._lut[c] = list(c2) + [1.]
        else:
            colors.append(c3)
            cmap._lut[c] = list(c3) + [1.]
    cmap.colors = tuple(colors)

    cmap._lut[cmap._i_over] = [0., 0., 0., 0.]
    cmap._lut[cmap._i_under] = [0., 0., 0., 0.]
    cmap._lut[cmap._i_bad] = [0., 0., 0., 0.]
    return cmap


def mkNifti(arr, mask, im, nii=True):
    out_im = np.zeros(mask.size, dtype=arr.dtype)
    inds = np.where(mask)[0]
    out_im[inds] = arr
    if nii:
        out_im = out_im.reshape(im.shape)
        out_im = nib.Nifti1Image(out_im, affine=im.affine)
    return out_im


def correct(ps_, rs_, p_crit=1e-2):
    sig_bool, ps_corrected, _, _ = multipletests(ps_, alpha=p_crit, method='fdr_by')
    indices = np.where(sig_bool)[0]
    return sig_bool, rs_[indices].min()


def filter_r(rs, ps):
    ps, threshold = correct(ps, rs)
    ps = np.invert(ps)
    indices = np.where(ps)[0]
    rs[indices] = 0.
    rs_mask = np.copy(rs)
    rs_mask[rs != 0.] = 1.
    return rs, rs_mask, threshold


class PlotEncoding():
    def __init__(self, args):
        self.process = 'PlotEncoding'
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.stat_dir = args.stat_dir
        self.mask_dir = args.mask_dir
        self.annotation_dir = args.annotation_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)
        self.features = []
        self.separate_features = args.separate_features
        self.overall_prediction = args.overall_prediction
        if not self.separate_features:
            if self.overall_prediction:
                self.cmap = sns.color_palette('magma', as_cmap=True)
                self.out_name = f'{self.figure_dir}/sub-{self.sid}_overall.png'
                self.threshold = None
            else:
                self.cmap = mk_cmap()
                self.out_name = f'{self.figure_dir}/sub-{self.sid}_grouped.png'
                self.threshold = 1.
        else:
            self.cmap = sns.color_palette('Paired', as_cmap=True)
            self.out_name = f'{self.figure_dir}/sub-{self.sid}_separate.png'
            self.threshold = 1.
        print(self.out_name)


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

    def plot_surface_stats(self, texture,
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
                                    total=len(modes) * len(hemis)):
            bg_map = self.fsaverage['sulc_%s' % hemi]
            ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
            axes.append(ax)
            plot_surf_roi(surf_mesh=self.fsaverage[f'infl_{hemi}'],
                          roi_map=texture[hemi],
                          view=mode, hemi=hemi,
                          bg_map=bg_map,
                          alpha=0.5,
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

    def load_features(self):
        df = pd.read_csv(f'{self.annotation_dir}/annotations.csv')
        train = pd.read_csv(f'{self.annotation_dir}/train.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        df.drop(columns=['video_name'], inplace=True)
        self.features = np.array(df.columns)

    def load(self):
        mask_im = nib.load(f'{self.mask_dir}/sub-all_stat-rho_statmap.nii.gz')
        mask = np.load(f'{self.mask_dir}/sub-all_reliability-mask.npy')

        if not self.overall_prediction:
            rs = np.load(f'{self.stat_dir}/sub-all/sub-all_feature-all_rs-filtered.npy').astype('bool')
            rs = mkNifti(rs, mask, mask_im, nii=False)

            base = f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-XXX_rs-mask.npy'
            pred = []
            for feature in self.features:
                arr = np.load(base.replace('XXX', feature))
                arr = np.expand_dims(arr, axis=1)
                if type(pred) is list:
                    pred = arr
                else:
                    pred = np.hstack([pred, arr])
            preference = np.argmax(pred, axis=1)
            # Make the argmax indexed at 1
            preference += 1

            # Make the preference values into a volume mask
            volume = mkNifti(preference, mask, mask_im, nii=False)
            volume[~rs] = 0
            volume = volume.astype('float')
            volume = nib.Nifti1Image(volume.reshape(mask_im.shape), affine=mask_im.affine)
        else:
            rs = np.load(f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-all_rs.npy')
            ps = np.load(f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-all_ps.npy')

            #Filter the r-values, set threshold, and save output
            rs, rs_mask, threshold = filter_r(rs, ps)
            self.threshold = threshold
            np.save(f'{self.stat_dir}/sub-{self.sid}/sub-all_feature-all_rs-filtered.npy', rs)
            np.save(f'{self.stat_dir}/sub-{self.sid}/sub-all_feature-all_rs-mask.npy', rs_mask)
            volume = mkNifti(rs, mask, mask_im)

        texture = {'left': self.vol_to_surf(volume, 'left'),
                   'right': self.vol_to_surf(volume, 'right')}
        return volume, texture

    def run(self):
        # load reliability files
        self.load_features()
        volume, texture = self.load()
        self.plot_surface_stats(texture, cmap=self.cmap,
                                output_file=self.out_name,
                                threshold=self.threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--separate_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overall_prediction', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mask_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/Reliability')
    parser.add_argument('--annotation_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw/annotations')
    parser.add_argument('--stat_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/VoxelPermutation')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PlotEncoding(args).run()


if __name__ == '__main__':
    main()
