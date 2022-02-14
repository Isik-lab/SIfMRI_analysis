import argparse
import os
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.manifold import MDS
from scipy import stats
from statsmodels.stats.multitest import multipletests

from itertools import combinations, chain

class feature_correlations():
    def __init__(self, args):
        self.process = 'feature_correlations'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def run(self, context='talk'):
        def diag(arr, cut=True):
            arr = np.tril(arr, -1)
            arr[arr == 0] = 'NaN'
            if cut:
                return arr[1:,:-1]
            else:
                return arr

        if context == 'talk':
            r_size = 12
        elif context == 'poster':
            r_size=18

        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        df = df.drop(columns=['video_name'])
        nqs = len(df.columns)
        ratings = np.array(df)
        ticks = df.columns
        rsm, ps_raw = stats.spearmanr(ratings)
        plotting_rsm = rsm
        plotting_rsm = diag(plotting_rsm)

        ## Correct p-values for multiple comparisons
        inds = np.tril(np.ones_like(ps_raw), -1)
        flat = ps_raw[inds == 1]
        _, ps_corrected, _, _ = multipletests(flat, method='fdr_by')
        ps_raw[inds == 1] = ps_corrected
        ps = diag(ps_raw, cut=True)

        sns.set(rc={'figure.figsize':(9,7)}, context=context)
        fig, ax = plt.subplots()
        cmap = cm.get_cmap(sns.diverging_palette(210, 15, s=90, l=40, n=11, as_cmap=True))
        # cmap = cm.get_cmap(sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True))
        cmap.set_bad('white')

        # # Make insignificant values white
        # for ((j,i),label) in np.ndenumerate(plotting_rsm):
        #     if ps[j,i] > 0.05:
        #         plotting_rsm[j,i] = np.nan

        im = plt.imshow(plotting_rsm, cmap=cmap, vmin=-.7, vmax=.7)
        for ((j,i),label) in np.ndenumerate(plotting_rsm):
            if not np.isnan(plotting_rsm[j, i]):
                color = 'black' if ps[j,i] < 0.05 else 'white'
                weight = 'bold' if ps[j,i] < 0.05 else 'normal'
                label = label if np.round_(label,decimals=1) != 0 else int(0)
                ax.text(i,j,'{:.1f}'.format(label), ha='center', va='center',
                        color=color, fontsize=14, weight=weight)
        cbar = plt.colorbar()
        cbar.ax.tick_params(size=0)

        # ax.set_yticks(np.arange(10), df.columns)
        ax.set_yticks(np.arange(nqs-1))
        ax.set_yticklabels(ticks[1:])
        ax.set_xticks(np.arange(nqs-1))
        ax.set_xticklabels(ticks[:-1], rotation=90,ha='center')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/correlation_matrix.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    parser.add_argument('--figure_dir', '-figures', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/figures')
    args = parser.parse_args()
    times = feature_correlations(args).run()

if __name__ == '__main__':
    main()
