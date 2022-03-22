#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from src.tools import permutation_test


def diag(arr, cut=True):
    arr = np.tril(arr, -1)
    arr[arr == 0] = 'NaN'
    if cut:
        return arr[1:, :-1]
    else:
        return arr


class FeatureCorrelations():
    def __init__(self, args):
        self.process = 'FeatureCorrelations'
        self.data_dir = args.data_dir
        self.set = args.set
        self.n_perm = args.n_perm
        self.plot_dists = args.plot_dists
        self.precomputed = args.precomputed
        self.rsa = args.rsa
        if self.rsa:
            self.H0 = 'greater'
        else:
            self.H0 = 'two_tailed'
        self.out_dir = f'{args.out_dir}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
            os.mkdir(f'{self.figure_dir}/dists_rsa-{self.rsa}')
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def plotting_dists(self, r, p, r_dist, name):
        _, ax = plt.subplots()
        sns.histplot(r_dist, element="step")
        ys = np.arange(0, self.n_perm / 5)
        xs = np.ones_like(ys) * r
        ax.plot(xs, ys, '--r')
        lim = np.abs(r) + .1
        plt.xlim([lim * -1, lim])
        plt.title(f'r = {r:.3f}, p = {p:.5f}')
        plt.savefig(f'{self.figure_dir}/dists_rsa-{self.rsa}/{name}.pdf')
        plt.close()

    def pairwise_coor(self, mat, correct=True):
        ps = []
        d = mat.shape[-1]
        a = np.ones((d, d), dtype='bool')
        rows, cols = np.where(np.tril(a, -1))
        count = 0
        for i, j in tqdm(zip(rows, cols), total=len(rows)):
            if self.rsa:
                test_inds = None
            else:
                test_inds = np.arange(mat[:, i].size)
            r, p, r_dist = permutation_test(mat[:, i], mat[:, j],
                                            n_perm=self.n_perm,
                                            test_inds=test_inds,
                                            H0=self.H0,
                                            rsa=self.rsa)
            if self.plot_dists:
                count += 1
                self.plotting_dists(r, p, r_dist, str(count).zfill(2))
            ps.append(p)
        ps = np.array(ps)
        if correct:
            ps, _, _, _ = multipletests(ps, method='fdr_bh')
        return ps

    def compute_mat(self, ratings):
        rsm, _ = spearmanr(ratings)
        ps = self.pairwise_coor(ratings)
        plotting_rsm = diag(rsm)
        p_bool = np.zeros_like(plotting_rsm, dtype='bool')
        i, j = np.where(~np.isnan(plotting_rsm))
        p_bool[i, j] = ps
        return plotting_rsm, p_bool

    def plot(self, rs, ps, ticks, context='talk'):
        if context == 'talk':
            r_size = 12
        else:
            r_size = 18

        nqs = len(ticks)
        sns.set(rc={'figure.figsize': (9, 7)}, context=context)
        fig, ax = plt.subplots()

        vmax = np.nanmax(np.abs(rs))
        if self.rsa:
            vmin = 0
            cmap = cm.get_cmap(sns.color_palette("light:b", as_cmap=True))
        else:
            vmin = -1 * vmax
            cmap = cm.get_cmap(sns.diverging_palette(210, 15, s=90, l=40, n=11, as_cmap=True))
        cmap.set_bad('white')

        plt.imshow(rs, cmap=cmap, vmin=vmin, vmax=vmax)
        for ((j, i), label) in np.ndenumerate(rs):
            if not np.isnan(rs[j, i]):
                color = 'black' if ps[j, i] else 'white'
                weight = 'bold' if ps[j, i] else 'normal'
                label = label if np.round_(label, decimals=1) != 0 else int(0)
                ax.text(i, j, '{:.1f}'.format(label), ha='center', va='center',
                        color=color, fontsize=r_size, weight=weight)
        ax.grid(False)
        cbar = plt.colorbar()
        cbar.ax.tick_params(size=0)

        ax.set_yticks(np.arange(nqs - 1))
        ax.set_yticklabels(ticks[1:])
        ax.set_xticks(np.arange(nqs - 1))
        ax.set_xticklabels(ticks[:-1], rotation=90, ha='center')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/correlation-matrix_rsa-{self.rsa}_set-{self.set}.pdf')
        plt.close()

    def save(self, arr, name):
        np.save(f'{self.out_dir}/{self.process}/{name}_rsa-{self.rsa}_set-{self.set}.npy', arr)

    def load_annotations(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        df = df.drop(columns=['video_name'])
        return df

    def run(self, context='talk'):
        if self.rsa:
            df = pd.read_csv(f'{self.out_dir}/FeatureRDMs/rdms_set-{self.set}.csv')
        else:
            df = self.load_annotations()

        if not self.precomputed:
            rs, ps = self.compute_mat(np.array(df))
            self.save(rs, 'rs')
            self.save(ps, 'ps')
        else:
            rs = np.load(f'{self.out_dir}/{self.process}/rs_rsa-{self.rsa}_set-{self.set}.npy')
            ps = np.load(f'{self.out_dir}/{self.process}/ps_rsa-{self.rsa}_set-{self.set}.npy')
        self.plot(rs, ps, df.columns, context=context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_perm', type=int, default=int(5e3))
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--plot_dists', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--rsa', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--precomputed', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    FeatureCorrelations(args).run()


if __name__ == '__main__':
    main()
