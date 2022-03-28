#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd

import nibabel as nib
import src.custom_plotting as cm
from src.tools import bootstrap
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


class ROIEncoding():
    def __init__(self, args):
        self.process = 'ROIEncoding'
        self.control = args.control
        self.roi = args.roi
        assert self.roi is not None, "must define roi"
        self.set = args.set
        self.n_samples = args.n_samples
        self.precomputed = args.precomputed
        self.pca_before_regression = args.pca_before_regression
        self.n_subjects = 4
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}/{self.control}'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/dists').mkdir(parents=True, exist_ok=True)
        self.reliability_mask = np.load(f'{self.out_dir}/Reliability/sub-all_reliability-mask.npy').astype('bool')
        self.roi_mask = self.load_roi_mask()

    def mask_array(self, a, b, dim=None):
        if type(a) is str:
            a = np.load(a)
        if type(b) is str:
            b = np.load(b).astype('bool')

        if len(a.shape) > 1:
            if dim < 0:
                out = a[..., b]
            else:  # dim > 0
                out = a[b, ...]
        else:
            out = a[b]
        return out

    def load_roi_mask(self):
        roi_mask = dict()
        for sid_ in range(1, self.n_subjects + 1):
            sid = str(sid_).zfill(2)
            file = glob.glob(f'{self.data_dir}/ROI_masks/sub-{sid}/sub-{sid}_*{self.roi}*nooverlap.nii.gz')
            cur = np.array(nib.load(file[0]).dataobj, dtype='bool').flatten()
            # Filter to the reliability mask because encoding was only done in reliable voxels
            roi_mask[f'sub-{sid}'] = self.mask_array(cur, self.reliability_mask)
        return roi_mask

    def load_encoding_results(self, sid):
        base = f'{self.out_dir}/VoxelEncoding/sub-{sid}_by_feature-True_control-{self.control}_pca_before_regression-{self.pca_before_regression}_XXX.npy'
        test_inds = np.load(base.replace('XXX', 'indices'))
        y_pred = self.mask_array(base.replace('XXX', 'y_pred'), self.roi_mask[f'sub-{sid}'], dim=-1)
        y_true = self.mask_array(base.replace('XXX', 'y_true'), self.roi_mask[f'sub-{sid}'], dim=-1)
        return y_pred.mean(axis=-1), y_true.mean(axis=-1), test_inds

    def load_permutation_results(self, sid, feature):
        base = f'{self.out_dir}/VoxelPermutation/sub-{sid}_feature-{feature}_control-{self.control}_pca_before_regression-{self.pca_before_regression}_XXX.npy'
        rs = self.mask_array(base.replace('XXX', 'rs'), self.roi_mask[f'sub-{sid}'])
        ps = self.mask_array(base.replace('XXX', 'ps'), self.roi_mask[f'sub-{sid}'])
        rs_null = self.mask_array(base.replace('XXX', 'rs-nulldist'), self.roi_mask[f'sub-{sid}'], dim=-1)
        return rs.mean(), ps.mean(), rs_null.mean(axis=-1)

    def load_features(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        return df.drop('video_name', axis=1).columns.to_numpy()

    def plotting_dists(self, r, p, r_dist, name):
        _, ax = plt.subplots()
        sns.histplot(r_dist, element="step")
        ys = np.arange(0, (float(r_dist.size) / 10.))
        xs = np.ones_like(ys) * r
        ax.plot(xs, ys, '--r')
        lim = np.abs(r) + .1
        plt.xlim([lim * -1, lim])
        plt.title(f'r = {r:.3f}, p = {p:.5f}')
        plt.savefig(f'{self.figure_dir}/dists/{name}.pdf')
        plt.close()

    def reorganize_data(self, features):
        categories = cm.feature_categories()
        df = pd.DataFrame()
        rs = np.ones((self.n_subjects, len(features)))
        rs_null = np.ones((self.n_subjects, len(features), self.n_samples))
        rs_var = np.ones_like(rs_null)
        for sid_ in range(self.n_subjects):
            sid = str(sid_ + 1).zfill(2)
            y_pred, y_true, test_inds = self.load_encoding_results(sid)
            for ifeature, feature in enumerate(features):
                r, p, r_null = self.load_permutation_results(sid, feature.replace(' ', '_'))
                r_var = bootstrap(y_pred[ifeature, :], y_true, test_inds,
                                  n_samples=self.n_samples)
                f = pd.DataFrame({'Subjects': [f'sub-{sid}'],
                                  'Features': [feature],
                                  'Feature category': [categories[feature]],
                                  'Pearson r': [r],
                                  'p value': [p],
                                  'low sem': [r - r_var.std()],
                                  'high sem': [r + r_var.std()],
                                  'Explained variance': [r ** 2]})
                # self.plotting_dists(r, p, r_null, f'sub-{sid}_roi-{self.roi}_feature-{feature}')
                df = pd.concat([df, f])
                rs[sid_, ifeature] = r
                rs_null[sid_, ifeature, :] = r_null
                rs_var[sid_, ifeature, :] = r_var
        df['Feature category'] = pd.Categorical(df['Feature category'],
                                                categories=['scene', 'object', 'social primitive', 'social'],
                                                ordered=True)
        return df, rs, rs_null.mean(axis=0), rs_var.mean(axis=0)

    def group_p(self, df, rs, rs_null, features):
        ps = []
        df['group_p_value'] = 0
        for i, feature in enumerate(features):
            p = np.sum(rs_null[i, :] > rs[i]) / rs_null.shape[-1]
            ps.append(p)
            df.loc[df.Features == feature, 'group_p_value'] = p
            self.plotting_dists(rs[i], p, rs_null[i, :], f'sub-all_roi-{self.roi}_feature-{feature}')
        _, ps_corrected, _, _ = multipletests(ps, method='fdr_by')

        for i, feature in enumerate(features):
            df.loc[df.Features == feature, 'group_pcorrected'] = ps_corrected[i]
        df['group sig'] = df['group_pcorrected'] < 0.05
        return df

    def ROI_noise_ceiling(self):
        split_half = self.mask_array(f'{self.out_dir}/Reliability/sub-all_stat-rho_statmap.npy',
                                        self.reliability_mask)
        noise_ceiling = 0
        for sid_ in range(1, self.n_subjects+1):
            sid = str(sid_).zfill(2)
            noise_ceiling += self.mask_array(split_half, self.roi_mask[f'sub-{sid}']).mean()
        return noise_ceiling / self.n_subjects

    def run(self):
        if not self.precomputed:
            features = self.load_features()
            df, rs, rs_null, rs_var = self.reorganize_data(features)
            df = self.group_p(df, rs.mean(axis=0), rs_null, features)
            df.to_csv(f'{self.out_dir}/{self.process}/{self.roi}.csv', index=False)
        else:
            df = pd.read_csv(f'{self.out_dir}/{self.process}/{self.roi}.csv')
        noise_ceiling = self.ROI_noise_ceiling()
        print(noise_ceiling)
        cm.plot_ROI_results(df, f'{self.figure_dir}/{self.roi}.pdf', 'Pearson r', noise_ceiling)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', type=str, default='pSTS')
    parser.add_argument('--control', type=str, default='none')
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--precomputed', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pca_before_regression', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    ROIEncoding(args).run()


if __name__ == '__main__':
    main()
