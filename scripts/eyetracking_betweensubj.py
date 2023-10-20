#
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from src.tools import corr, calculate_p, bootstrap, perm
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
matplotlib.use('TkAgg')


def q_map(q):
    if 0.01 < q < 0.05:
        out = '*'
    elif 0.001 < q < 0.01:
        out = '**'
    elif q < 0.001:
        out = '***'
    else:
        out = ''
    return out


def load_feature_corr(top_dir, subjs):
    df = []
    for subj in subjs:
        subj_str = str(subj).zfill(3)
        subj_df = pd.read_pickle(f'{top_dir}subj{subj_str}.pkl')
        subj_df['subj'] = subj
        df.append(subj_df)
    return pd.concat(df)


def group_analysis(df):
    avg_df = []
    ps = []
    for feature, cur in df.groupby('feature'):
        null = np.nanmean(np.vstack(cur.r_null.values), axis=0)
        var = np.nanmean(np.vstack(cur.r_var.values), axis=0)
        r = cur.r.mean()
        p = calculate_p(null, r, n_perm_=len(null), H0_='two_tailed')
        ci_low, ci_high = np.nanpercentile(var, [2.5, 97.5])
        avg_df.append({'feature': feature, 'r': r, 'p': p,
                       'ci_low': ci_low, 'ci_high': ci_high})
        ps.append(p)
    avg_df = pd.DataFrame(avg_df)

    bools, qs = fdrcorrection(ps)
    avg_df['sig'] = bools
    avg_df['q'] = qs
    avg_df['sig_text'] = [q_map(q) for q in qs]
    avg_df.replace({'transitivity': 'object', 'expanse': 'spatial expanse'}, inplace=True)
    df.replace({'transitivity': 'object', 'expanse': 'spatial expanse'}, inplace=True)
    return df, avg_df.set_index('feature')


def feature2color(key=None):
    d = dict()
    d['indoor'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['spatial expanse'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['object'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['agent distance'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['facingness'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['joint action'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['communication'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['valence'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    d['arousal'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    if key is not None:
        return d[key]
    else:
        return d


def plot_feature_corr(df, p_df, title=None, out_dir=None):
    print(out_dir)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='paper', style='white', rc=custom_params)
    _, ax = plt.subplots(figsize=(3, 2.5))
    sns.barplot(x='feature', y='r', data=df,
                errorbar=None,
                zorder=1, color='gray')
    sns.stripplot(x='feature', y='r', data=df,
                  zorder=3, color='black', jitter=.35)
    for scatters in ax.collections:
        scatters.set_sizes([5])
    ymin, ymax = ax.get_ylim()
    for x, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        ax.text(x, ymax, p_df.loc[label._text, 'sig_text'],
                fontsize=10,horizontalalignment='center')
        ax.vlines(x, ymin=p_df.loc[label._text, 'ci_low'],
                  ymax=p_df.loc[label._text, 'ci_high'],
                  color='black', linewidth=1, zorder=2)

    for bar, label in zip(ax.patches, ax.get_xticklabels()):
        feature = label._text
        color = feature2color(feature)
        bar.set_color(color)
    ax.set_ylim([ymin, ymax+(ymax*.2)])
    ax.set_xlabel('')
    ax.set_ylabel('Correlation ($r$)')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.title(title)
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir)
    plt.close('all')


class EyeTracking_BetweenSubj:
    def __init__(self, args):
        self.process = 'EyeTracking_BetweenSubj'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        self.subjs = [4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
        self.features = ['indoor', 'expanse', 'transitivity',
                         'facingness', 'agent distance',
                         'joint action', 'communication',
                         'valence', 'arousal']

    def load_heatmaps(self):
        subj_data = dict()
        for subj in self.subjs:
            subj_str = str(subj).zfill(3)
            df = pd.read_csv(f'{self.out_dir}/EyeTracking_WithinSubj/heatmaps/subj{subj_str}.csv')
            subj_data[f'subj{subj_str}'] = df.groupby(['video_name']).mean(numeric_only=True)
        return subj_data

    def intrasubj_reliability(self):
        df = []
        for subj in self.subjs:
            subj_str = str(subj).zfill(3)
            subj_df = pd.read_csv(f'{self.out_dir}/EyeTracking_WithinSubj/intrasubj_reliability/subj{subj_str}.csv')
            subj_df['subj'] = f'subj{subj_str}'
            df.append(subj_df)
        df = pd.concat(df)
        df.to_csv(f'{self.out_dir}/{self.process}/intrasubj_reliability.csv', index=False)
        avg_df = df.groupby("subj").mean(numeric_only=True)
        print(f'average intrasubject reliability: {avg_df.r.mean()}')
        print(f'intrasubject reliability range: {avg_df.r.min()} to {avg_df.r.max()}')

    def compute_intersubj(self, subj_data):
        columns = None
        out_df = []
        for subj in self.subjs:
            subj_str = str(subj).zfill(3)
            if not columns:
                columns = [col for col in subj_data[f'subj{subj_str}'].columns if 'heatmap' in col]
                videos = subj_data[f'subj{subj_str}'].index.to_list()
            loo_data = subj_data[f'subj{subj_str}']
            other_subjs = {key: subj_data[key] for key in subj_data.keys() if not subj_str in key}
            other_data = pd.concat(other_subjs).reset_index(drop=False).groupby('video_name').mean(numeric_only=True)
            for vid in videos:
                x = loo_data.loc[vid, columns].to_numpy().astype('float')
                y = other_data.loc[vid, columns].to_numpy().astype('float')
                out_df.append({'loo': f'subj{subj_str}', 'video_name': vid, 'r': corr(x, y)})
        df = pd.DataFrame(out_df)
        df.to_csv(f'{self.out_dir}/{self.process}/intersubj_reliability.csv', index=False)
        avg_df = df.groupby('loo').mean(numeric_only=True)
        print(f'average intersubject reliability: {avg_df.r.mean()}')
        print(f'intersubject reliability range: {avg_df.r.min()} to {avg_df.r.max()}')
        return df

    def plot_intersubj(self, df):
        _, ax = plt.subplots(2)
        video = df.groupby('video_name').mean(numeric_only=True).reset_index(drop=False)
        ax[0].hist(video.r)
        ax[0].set_title('video')
        ax[0].vlines(x=video.mean(numeric_only=True).r, ymin=0, ymax=9, colors='r')
        loo = df.groupby('loo').mean(numeric_only=True).reset_index(drop=False)
        ax[1].hist(loo.r)
        ax[1].set_title('subj')
        ax[1].vlines(x=loo.mean(numeric_only=True).r, ymin=0, ymax=9, colors='r')
        plt.savefig(f'{self.figure_dir}/hist.pdf')

    def load_ratings(self):
        annotations = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        annotations.drop(columns=['dominance', 'cooperation', 'intimacy'], inplace=True)
        test = pd.read_csv(f'{self.data_dir}/annotations/test.csv')
        return test.merge(annotations, on='video_name', how='left').reset_index(drop=True)

    def intersubj_annotation_corr(self, df, out_file):
        annotations = self.load_ratings()
        subjs = df.loo.unique()
        df_indexed = df.sort_values(by=['loo', 'video_name']).set_index(['loo', 'video_name'])
        out_df = []
        for subj in tqdm(subjs):
            subj_df = df_indexed.loc[subj]
            merged = subj_df.merge(annotations, on='video_name', how='left')
            for feature in self.features:
                r, p, r_null = perm(merged[feature].to_numpy(), merged['r'].to_numpy(),
                                    H0='two_tailed', square=False, verbose=False)
                r_var = bootstrap(merged[feature].to_numpy(), merged['r'].to_numpy(),
                                   square=False, verbose=False)
                out_df.append({'subj': subj, 'feature': feature, 'r': r, 'p': p,
                               'r_null': r_null, 'r_var': r_var})
        out_df = pd.DataFrame(out_df)
        out_df.to_pickle(out_file)

    def intersubj_reliability(self):
        subj_data = self.load_heatmaps()
        df = self.compute_intersubj(subj_data)
        self.plot_intersubj(df)

        intersubj_feature_file = f'{self.out_dir}/{self.process}/intersubj_feature.pkl'
        if not os.path.exists(intersubj_feature_file):
            self.intersubj_annotation_corr(df, intersubj_feature_file)

    def social_scene_looking(self, df):
        communication = df.loc[df.feature == 'communication'].set_index('subj')
        expanse = df.loc[df.feature == 'spatial expanse'].set_index('subj')
        null = np.abs(np.vstack(expanse.r_null.values)) - np.abs(np.vstack(communication.r_null.values))
        avg_null = null.mean(axis=0)
        r_diff = np.abs(expanse.r) - np.abs(communication.r)
        avg_r_diff = r_diff.mean()
        print(f'spatial expanse and communciation difference = {avg_r_diff:.2f}')
        return calculate_p(avg_null, avg_r_diff, n_perm_=5000, H0_='greater')

    def group_analyses(self):
        for analysis, title in zip(['reliability_feature', 'face_feature', 'right_feature', 'intersubj_feature'],
                                ['Reliability-feature', 'Face looking', 'Right looking', 'Intersubj-feature']):
            print(analysis)
            if analysis != 'intersubj_feature':
                df = load_feature_corr(f'{self.out_dir}/EyeTracking_WithinSubj/{analysis}/', self.subjs)
            else:
                df = pd.read_pickle(f'{self.out_dir}/{self.process}/intersubj_feature.pkl')
            df, avg_df = group_analysis(df)

            if analysis == 'face_feature':
                p = self.social_scene_looking(df)
                print(f'spatial expanse is more related to face looking than communication (p = {p:.4f})')

            avg_df.to_csv(f'{self.out_dir}/{self.process}/{analysis}.csv')
            print(analysis)
            plot_feature_corr(df, avg_df, title=title, out_dir=f'{self.figure_dir}/{analysis}.svg')

    def run(self):
        self.intersubj_reliability()
        self.intrasubj_reliability()
        self.group_analyses()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    EyeTracking_BetweenSubj(args).run()


if __name__ == '__main__':
    main()
