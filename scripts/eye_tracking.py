#
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import mmcv
from PIL import Image
from scipy.stats import pearsonr
from itertools import product

matplotlib.use('TkAgg')


def plt_fixation(image, heatmap, title, out_name):
    _, ax = plt.subplots()
    ax.imshow(image, zorder=1)
    ax.imshow(heatmap, zorder=2, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close('all')


def generate_screen_image(image, x_max=1600, y_max=900):
    old_size = (y_max, y_max)
    new_size = (x_max, y_max)
    smaller_image = image.resize(old_size)
    new_im = Image.new("RGB", (x_max, y_max), (175, 175, 175))
    box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
    new_im.paste(smaller_image, box)
    return new_im


def generate_heatmap(xs, ys, x_max=1600, y_max=900, scaling_factor=100, gen_img=False):
    x_bins, y_bins = int(x_max / scaling_factor), int(y_max / scaling_factor)
    if np.sum(np.isnan(ys)) > (len(ys) * .3):
        heatmap = np.full([y_bins, x_bins], np.nan)
    else:
        heatmap, _, _ = np.histogram2d(ys, xs,
                                       bins=[y_bins, x_bins],
                                       range=[[0, y_max], [0, x_max]],
                                       density=True)

    if gen_img:
        im = Image.fromarray(np.uint8(cm.OrRd(heatmap) * 255))
        im = im.resize((x_max, y_max), resample=Image.NEAREST)
    else:
        im = None
    return heatmap, im


def split_half_correlation(x):
    even = x.loc[x.even, 'heatmap'].to_numpy()[0]
    odd = x.loc[x.even == False, 'heatmap'].to_numpy()[0]
    if not np.all(np.isnan(even)) and not np.all(np.isnan(odd)):
        return pearsonr(even, odd).statistic
    else:
        return np.nan


class EyeTracking:
    def __init__(self, args):
        self.process = 'EyeTracking'
        self.sid = str(int(args.s_num)).zfill(3)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.figure_dir}/heatmaps/subj{self.sid}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/{self.process}/intersubj_reliability').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/{self.process}/intersubj_feature_corr').mkdir(parents=True, exist_ok=True)

    def load_data(self):
        df = pd.read_csv(f'{self.data_dir}/eyetracking/subj{self.sid}.csv')
        df = df.loc[df.condition == 1]
        return df.drop(columns=['condition']).reset_index(drop=True)

    def load_frame(self, video, frame_num=0):
        in_file = f'{self.data_dir}/videos/{video}'
        video_obj = mmcv.VideoReader(in_file)
        frame = video_obj[frame_num]
        frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return generate_screen_image(frame_rgb)

    def gen_heatmaps(self, df):
        heatmap_df = []
        for (video, block, run), rep_df in df.groupby(['video_name', 'run', 'block']):
            single_heatmap, _ = generate_heatmap(rep_df.x, rep_df.y)
            heatmap_df.append({'video_name': video, 'block': block, 'run': run,
                               'heatmap': single_heatmap.flatten()})
        heatmap_df = pd.DataFrame(heatmap_df)
        heatmap_df['repetition'] = heatmap_df.groupby(['video_name']).cumcount()
        average_heatmap = heatmap_df.groupby('video_name').apply(lambda x: np.nanmean(np.vstack(x.heatmap.to_list()), axis=0)).reset_index(drop=False, name='heatmap')
        heatmap_df['even'] = False
        heatmap_df.loc[heatmap_df.repetition % 2 == 0, 'even'] = True
        split_half = heatmap_df.groupby(['even', 'video_name']).apply(lambda x: np.nanmean(np.vstack(x.heatmap.to_list()), axis=0)).reset_index(drop=False, name='heatmap')
        return average_heatmap, split_half

    def intersubj_reliability(self, split_half):
        reliability = split_half.groupby('video_name').apply(lambda x: split_half_correlation(x)).reset_index(
            drop=False, name='correlation')
        reliability.to_csv(f'{self.out_dir}/{self.process}/intersubj_reliability/subj{self.sid}.csv', index=False)
        return reliability

    def load_ratings(self):
        annotations = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        annotations.drop(columns=['dominance', 'cooperation', 'intimacy'], inplace=True)
        test = pd.read_csv(f'{self.data_dir}/annotations/test.csv')
        return test.merge(annotations, on='video_name', how='left').reset_index(drop=True)

    def reliability_rating_correlation(self, reliability, annotations):
        merged = reliability.merge(annotations, on='video_name', how='left')
        features = merged.columns.drop(['video_name', 'correlation'])
        out_df = []
        for feature in features:
            r = pearsonr(merged[feature], merged['correlation']).statistic
            out_df.append({'feature': feature, 'r': r})
            print(feature, r)
        out_df = pd.DataFrame(out_df)
        out_df.to_csv(f'{self.out_dir}/{self.process}/intersubj_feature_corr/subj{self.sid}.csv', index=False)

    def run(self):
        df = self.load_data()
        average_heatmap, split_half = self.gen_heatmaps(df)
        reliability = self.intersubj_reliability(split_half)
        annotations = self.load_ratings()
        self.reliability_rating_correlation(reliability, annotations)


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
    EyeTracking(args).run()


if __name__ == '__main__':
    main()
