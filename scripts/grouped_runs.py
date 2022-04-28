#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path

class GroupRuns():
    def __init__(self, args):
        self.sid = str(args.s_num).zfill(2)
        self.process = 'GroupRuns'
        self.set = args.set
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}/sub-{self.sid}'
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        videos = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        videos.sort_values(by=['video_name'], inplace=True)

        all_runs = None
        print(f'sub-{self.sid}')
        for ivid in range(len(videos)):
            vid = videos.loc[ivid, 'video_name'].split('.')[0]
            print(f'{ivid}: {vid}')
            cond_files = sorted(glob.glob(f"{self.data_dir}/betas/sub-{self.sid}/*cond-{vid}*.npy"))

            cond_arr = None
            for irun in range(len(cond_files)):
                arr = np.load(cond_files[irun])
                if cond_arr is None:
                    cond_arr = np.zeros((len(cond_files), arr.size))

                cond_arr[irun, :] = arr.flatten()

            if all_runs is None:
                all_runs = np.zeros((len(videos), arr.size))
                odd = np.zeros_like(all_runs)
                even = np.zeros_like(all_runs)

            even[ivid, :] = cond_arr[1::2, :].mean(axis=0)
            odd[ivid, :] = cond_arr[::2, :].mean(axis=0)
            all_runs[ivid, :] = cond_arr.mean(axis=0)

        # Save the subject data
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.set}-even-data.npy', even)
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.set}-odd-data.npy', odd)
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.set}-data.npy', all_runs)
        print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    GroupRuns(args).run()

if __name__ == '__main__':
    main()


