#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


class fMRIBehaviorAnalysis:
    def __init__(self, args):
        self.process = 'fMRIBehaviorAnalysis'
        self.sid = str(args.s_num).zfill(2)
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        files = sorted(glob.glob(f'{self.data_dir}/sub-{self.sid}/*.tsv'))
        df = pd.DataFrame()
        for file in files:
            cur = pd.read_csv(file, sep='\t')
            df = pd.concat([df, cur])
        go = df.loc[df.trial_type == 'crowd', 'response'].to_numpy()
        no_go = df.loc[df.trial_type == 'dyad', 'response'].to_numpy()
        acc = np.sum((df.trial_type == 'crowd').to_numpy() == df.response.to_numpy().astype('bool'))/len(df)
        H = np.sum(go == 1) / len(go)
        FA = np.sum(no_go == 1) / len(no_go)
        d_prime = stats.norm.ppf(H) - stats.norm.ppf(FA)
        with open(f'{self.out_dir}/sub-{self.sid}_d-prime.txt', 'w') as f:
            f.write(f"Accuracy = {acc * 100:.3f}% \n")
            f.write(f"Hit rate = {H:.5f} \n")
            f.write(f"False alarm rate = {FA:.5f} \n")
            f.write(f"d' = {d_prime:.5f} \n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw/SIdyads_behavior')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    fMRIBehaviorAnalysis(args).run()


if __name__ == '__main__':
    main()
