# /Applications/anaconda3/envs/opencv/bin/python

import argparse
import imageio
import os
import numpy as np
import pandas as pd
import moten

class MotionEnergyActivations():
    def __init__(self, args):
        self.process = 'MotionEnergyActivations'
        self.set = args.set
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def run(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df.sort_values(by=['video_name'], inplace=True)
        vid_dir = f'{self.data_dir}/videos'

        # Create a pyramid of spatio-temporal gabor filters
        vdim, hdim, fps = 500, 500, 30
        pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=fps)

        out_moten = []
        for vid in df.video_name:
            vid_obj = imageio.get_reader(f'{vid_dir}/{vid}', 'ffmpeg')
            vid = []
            for i in range(90):
                vid.append(vid_obj.get_data(i).mean(axis=-1))
            vid = np.array(vid)
            moten_features = pyramid.project_stimulus(vid)
            out_moten.append(moten_features.mean(axis=0))
        out_moten = np.array(out_moten)
        np.save(f'{self.out_dir}/motion_energy_set-{self.set}.npy', out_moten)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    MotionEnergyActivations(args).run()

if __name__ == '__main__':
    main()
