#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd


class ActionCategories():
    def __init__(self, args):
        self.process = 'ActionCategories'
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def run(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/{self.dataset}_categories.csv')
        cats = df.groupby('action_categories', as_index=False).count()
        cats = cats.rename(columns={"video_name": "n"})

        with open(f'{self.figure_dir}/{self.dataset}_tolatex.txt', 'w') as f:
            for cat, n in zip(cats.action_categories, cats.n):
                if '+' in cat:
                    cat = cat.replace('+', ' ')
                line = (cat + ' & ' + str(n) + ' \\\ \n')
                print(line)
                f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    ActionCategories(args).run()


if __name__ == '__main__':
    main()
