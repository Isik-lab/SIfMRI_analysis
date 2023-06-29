#
import argparse
from itertools import product

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns


class FaceFeature:
    def __init__(self, args):
        self.process = 'FaceFeature'
        self.sid = str(int(args.s_num)).zfill(3)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)

    def load_ratings(self):
        annotations = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        annotations.drop(columns=['dominance', 'cooperation', 'intimacy'], inplace=True)
        test = pd.read_csv(f'{self.data_dir}/annotations/test.csv')
        return test.merge(annotations, on='video_name', how='left').reset_index(drop=True)

    def load_faces(self):
        faces = pd.read_csv(f'{self.data_dir}/face_annotation/bounding_boxes.csv')
        faces = faces[['video_name', 'frame', 'face', 'face_area', 'face_centrality']]
        faces = faces.groupby(['video_name', 'face']).mean(numeric_only=True).reset_index()
        faces = faces.groupby(['video_name']).mean(numeric_only=True).reset_index()
        faces.drop(columns=['frame'], inplace=True)
        test = pd.read_csv(f'{self.data_dir}/annotations/test.csv')
        return test.merge(faces, on='video_name', how='left').reset_index(drop=True)

    def face_feature_correlation(self, faces, annotations):
        merged = annotations.merge(faces, on='video_name', how='left').reset_index(drop=True)
        features = merged.columns.drop(['video_name', 'face_area', 'face_centrality'])
        out_df = []
        for stat, feature in product(['face_area', 'face_centrality'], features):
            r = pearsonr(merged[feature], merged[stat]).statistic
            out_df.append({'feature': feature, 'stat': stat, 'r': r})
        out_df = pd.DataFrame(out_df)
        out_df.to_csv(f'{self.out_dir}/{self.process}/face_annotation_correlation.csv', index=False)
        return out_df

    def plot_results(self, df):
        sns.barplot(x='feature', y='r', hue='stat', data=df)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/face_annotation_correlation.pdf')

    def run(self):
        annotations = self.load_ratings()
        faces = self.load_faces()
        df = self.face_feature_correlation(faces, annotations)
        self.plot_results(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default='2')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    FaceFeature(args).run()

if __name__ == '__main__':
    main()