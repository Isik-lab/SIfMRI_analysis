# /Applications/anaconda3/envs/opencv/bin/python

import argparse
import imageio
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class MotionEnergyActivations():
    def __init__(self, args):
        self.process = 'MotionEnergyActivations'
        self.set = args.set
        self.overwrite = args.overwrite
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

    def get_PCs(self, X):
        pca = PCA(whiten=False, n_components=20)
        pca.fit(X)
        _, axes = plt.subplots(2)
        axes[0].plot(pca.explained_variance_ratio_, '-o')
        axes[0].set_title('Explained variance')
        axes[1].plot(pca.explained_variance_ratio_.cumsum(), '-o')
        axes[1].set_title('Cumulative explained variance')
        plt.savefig(f'{self.figure_dir}/pca_visualization_set-{self.set}.pdf')

    def get_moten(self):
        # import moten
        df = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df.sort_values(by=['video_name'], inplace=True)
        vid_dir = f'{self.data_dir}/videos'

        # Create a pyramid of spatio-temporal gabor filters
        vdim, hdim, fps = 500, 500, 30
        # pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=fps)

        out_moten = []
        for ivid, vid in enumerate(df.video_name):
            print(f'{ivid}: {vid}')
            vid_obj = imageio.get_reader(f'{vid_dir}/{vid}', 'ffmpeg')
            vid = []
            for i in range(90):
                vid.append(vid_obj.get_data(i).mean(axis=-1))
            vid = np.array(vid)
            moten_features = pyramid.project_stimulus(vid)
            out_moten.append(moten_features.mean(axis=0))
        return np.array(out_moten)

    def run(self):
        out_file = f'{self.out_dir}/motion_energy_set-{self.set}.npy'
        if not os.path.exists(out_file) or self.overwrite:
            out_moten = self.get_moten()
            np.save(out_file, out_moten)
        else:
            out_moten = np.load(out_file)
        self.get_PCs(out_moten)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    MotionEnergyActivations(args).run()

if __name__ == '__main__':
    main()
