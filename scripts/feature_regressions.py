#
import argparse

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

from src.tools import perm
import seaborn as sns
from itertools import combinations

from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def scale(train_, test_):
    mean = np.nanmean(train_, axis=0).squeeze()
    variance = np.nanstd(train_, axis=0).squeeze()
    variance[np.isclose(variance, 0.)] = np.nan
    train_ = (train_ - mean) / variance
    return train_, (test_ - mean) / variance


def evaluate(y_hat_, y_test_):
    r, p, _ = perm(y_hat_, y_test_, square=False)
    return r[0], p[0]


def regression(X_train_, X_test_, y_train_, y_test_):
    X_train_scaled, X_test_scaled = scale(X_train_, X_test_)
    y_train_scaled, y_test_scaled = scale(y_train_, y_test_)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train_scaled, y_train_scaled)
    y_hat = lr.predict(X_test_scaled)
    return evaluate(y_hat, y_test_scaled)


def get_color(label):
    from src.custom_plotting import feature_colors, custom_palette
    colors = feature_colors()
    palette = custom_palette(rgb=False)
    out = 'gray'
    for key in colors.keys():
        if key in label:
            out = palette[colors[key]]
            break
    return out


class FeatureRegression:
    def __init__(self, args):
        self.process = 'FeatureRegression'
        self.sid = str(int(args.s_num)).zfill(3)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.regressors = ['alexnet', 'moten',
                           'indoor', 'expanse', 'transitivity',
                           'agent distance', 'facingness',
                           'joint action', 'communication', 'valence', 'arousal']
        self.tick_name = ['AlexNet-conv2', 'motion energy',
                          'indoor', 'expanse', 'object',
                          'agent distance', 'facingness',
                          'joint action', 'communication', 'valence', 'arousal']
        self.plotting_map = {old: new for old, new in zip(self.regressors, self.tick_name)}

    def load_ratings(self):
        annotations = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        annotations.drop(columns=['dominance', 'cooperation', 'intimacy'], inplace=True)
        return annotations

    def load_faces(self):
        # load all data
        # faces = pd.read_csv(f'{self.data_dir}/annotations/face_annotations.csv')
        annotations = self.load_ratings()
        moten = pd.read_csv(f'{self.out_dir}/ActivationPCA/moten_PCs.csv')
        alexnet = pd.read_csv(f'{self.out_dir}/ActivationPCA/alexnet_PCs.csv').drop(columns=['split'])

        # merge the dataframes
        # df = faces.merge(annotations, on='video_name')
        df = annotations.merge(moten, on='video_name')
        df = df.merge(alexnet, on='video_name')
        return df

    def pairwise_regression(self, df):
        regress_results = []
        ps = []
        for x_feature, y_feature in combinations(self.regressors, 2):
            print(x_feature, y_feature)
            x_columns = [col for col in df.columns if x_feature in col]
            y_columns = [col for col in df.columns if y_feature in col]
            X_train, X_test = df.loc[df.split == 'train', x_columns].to_numpy(), df.loc[
                df.split == 'test', x_columns].to_numpy()
            y_train, y_test = df.loc[df.split == 'train', y_columns].to_numpy(), df.loc[
                df.split == 'test', y_columns].to_numpy()
            if y_train.shape[-1] > 1:
                y_train = np.expand_dims(y_train.mean(axis=-1), axis=-1)
                y_test = np.expand_dims(y_test.mean(axis=-1), axis=-1)
            r, p = regression(X_train, X_test, y_train, y_test)
            regress_results.append({'x': x_feature,
                                    'y': y_feature,
                                    'r': r, 'r2': np.sign(r) * (r ** 2), 'p': p})
            ps.append(p)
        out_df = pd.DataFrame(regress_results)
        sig, q = fdrcorrection(ps)
        out_df['sig'] = sig
        out_df['q'] = q
        out_df.replace(self.plotting_map, inplace=True)
        cat_type = CategoricalDtype(categories=self.tick_name, ordered=True)
        out_df.x = out_df.x.astype(cat_type)
        out_df.y = out_df.y.astype(cat_type)
        return out_df

    def plot_results(self, out_df):
        square = out_df.pivot(index='y', columns='x', values='r2')
        qs = out_df.pivot(index='y', columns='x', values='q').to_numpy()  # .astype('bool')
        sig = qs < 0.05
        square_labels = np.ones_like(qs).astype('str')
        square_labels[np.invert(sig)] = ''
        vals = [f'{val:.1f}' for val in square.to_numpy()[sig]]
        square_labels[sig] = vals
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(context='paper', style='white', rc=custom_params)
        _, ax = plt.subplots(1, figsize=(3.5, 3))
        sns.heatmap(square, ax=ax,
                    annot=square_labels, fmt='',
                    annot_kws={"fontsize": 7},
                    linewidth=.5,
                    cmap=sns.color_palette("Blues", as_cmap=True),
                    cbar_kws={'label': 'Explained variance ($\mathit{r^2}$)'})
        ax.set(xlabel="", ylabel="")

        for pointer in ax.get_yticklabels():
            pointer.set_color(get_color(pointer._text))
            pointer.set_weight('bold')

        for pointer in ax.get_xticklabels():
            pointer.set_color(get_color(pointer._text))
            pointer.set_weight('bold')

        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/feature_regression.svg')

    def save_results(self, out_df):
        out_df.to_csv(f'{self.out_dir}/{self.process}/regression_results.csv', index=False)

    def run(self):
        df = self.load_faces()
        out_df = self.pairwise_regression(df)
        self.save_results(out_df)
        self.plot_results(out_df)


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
    FeatureRegression(args).run()


if __name__ == '__main__':
    main()
