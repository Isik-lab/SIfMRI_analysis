#
import argparse
import os

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from PIL import Image, ImageDraw
from tqdm import tqdm
from src.tools import corr, perm, bootstrap
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
matplotlib.use('TkAgg')


def plt_fixation(image, heatmap, bb1, bb2, out_name,
                 bb_color=(255, 255, 255), bb_width=5):
    image.paste(heatmap, (0, 0), heatmap)
    draw = ImageDraw.Draw(image)
    if bb1:
        draw.rectangle(bb1, outline=bb_color, width=bb_width)
    if bb2:
        draw.rectangle(bb2, outline=bb_color, width=bb_width)
    image.save(out_name)


def str2ints(box):
    box = box.strip('][').split(', ')
    return np.array(box).astype('int').tolist()


def generate_heatmap(xs, ys, x_max=1600, y_max=900,
                     border=350, scaling_factor=180,
                     n_samples=1500):
    x_frame_min = border + 1
    x_frame_max = x_max-border
    bool_inds = (xs <= x_frame_min) | (xs > x_frame_max)
    xs[bool_inds] = np.nan
    ys[bool_inds] = np.nan
    bins = int(y_max / scaling_factor)
    if np.sum(np.isnan(ys)) > (len(ys) * .3):
        heatmap = np.full([bins, bins], np.nan)
    else:
        heatmap, _, _ = np.histogram2d(ys, xs,
                                       bins=[bins, bins],
                                       range=[[0, y_max], [x_frame_min, x_frame_max]])
        heatmap /= np.sum(np.invert(np.isnan(xs)))
    return heatmap


def heatmap2img(heatmap, outsize, alpha=100):
    scaled_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    color_map = np.uint8(cm.magma(scaled_heatmap) * 255)
    color_map[:, :, -1] = alpha
    im = Image.fromarray(color_map)
    return im.resize(outsize, resample=Image.NEAREST)


def facemap2img(facemap, outsize):
    im = Image.fromarray(np.uint8(cm.gray(facemap) * 255))
    return im.resize(outsize, resample=Image.NEAREST)


def get_right_inds(x_dim):
    inds = np.arange(x_dim**2).reshape((20, 20))
    inds[:, :int(x_dim/2)] = 0
    return inds[inds != 0]


def right_looking(df, resolution):
    right_cols = np.array([f'heatmap{i}' for i in get_right_inds(resolution)])
    df['prop_right_looking'] = df[right_cols].sum(axis=1)
    return df


def face_looking(main_data, bounding_boxes, resolution):
    def prop_face(s, cols1, cols2):
        face_inds = np.where(s[cols1])[0]
        return s[cols2[face_inds]].sum()

    joined = main_data.merge(bounding_boxes,
                             how='left', on='video_name')

    print('calculating looking at faces...')
    face_cols = np.array([f'face{i}' for i in range(resolution**2)])
    heatmap_cols = np.array([f'heatmap{i}' for i in range(resolution**2)])
    joined[face_cols] = joined[face_cols].astype('bool')
    joined['prop_face_looking'] = joined.apply(prop_face, args=(face_cols, heatmap_cols), axis=1)
    print(f'proportion looking at faces = {joined.prop_face_looking.mean()}')
    return joined


def load_bounding_boxes(data_path, frame_size, resolution):
    def face_map(s, out_size):
        mat = np.zeros((out_size, out_size), dtype='bool')
        mat[s.top:s.bottom+1, s.left:s.right+1] = True
        mat = mat.flatten()
        for i in range(len(mat)):
            s[f'face{i}'] = mat[i]
        return s
    bb = pd.read_csv(f'{data_path}/face_annotation/bounding_boxes.csv')
    bb_average = bb[['video_name', 'face', 'top', 'left', 'bottom', 'right']]
    bb_average = bb_average.groupby(['video_name', 'face']).mean(numeric_only=True).reset_index(drop=False)
    bb_average[['top', 'left', 'bottom', 'right']] = bb_average[['top', 'left', 'bottom', 'right']] / (frame_size / resolution)
    bb_average[['top', 'left', 'bottom', 'right']] = bb_average[['top', 'left', 'bottom', 'right']].astype('int')
    bb_average = bb_average.apply(face_map, args=(resolution,), axis=1)
    return bb.set_index(['video_name', 'frame', 'face']),\
           bb_average.groupby('video_name').sum(numeric_only=False).reset_index(drop=False)


class EyeTracking_WithinSubj:
    def __init__(self, args):
        self.process = 'EyeTracking_WithinSubj'
        self.sid = str(int(args.s_num)).zfill(3)
        print(f'starting on subj{self.sid}')
        self.overwrite = args.overwrite
        self.save_heatmaps = args.save_heatmaps
        self.res = args.resolution
        self.frame_size = 500
        self.presentation_size = 900
        self.screen_width = 1600
        self.x_border = 350
        self.n_samples = 1500
        self.features = ['indoor', 'expanse', 'transitivity',
                         'facingness', 'agent distance',
                         'joint action', 'communication',
                         'valence', 'arousal']
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        for analysis in ['heatmaps', 'intrasubj_reliability',
                         'reliability_feature', 'face_feature', 'right_feature']:
            Path(f'{self.figure_dir}/{analysis}').mkdir(parents=True, exist_ok=True)
            Path(f'{self.out_dir}/{self.process}/{analysis}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/heatmaps/subj{self.sid}').mkdir(parents=True, exist_ok=True)

    def load_data(self):
        df = pd.read_csv(f'{self.data_dir}/eyetracking/subj{self.sid}.csv')
        df = df.loc[df.condition == 1]
        return df.drop(columns=['condition']).reset_index(drop=True)

    def load_ratings(self):
        annotations = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        annotations.drop(columns=['dominance', 'cooperation', 'intimacy'], inplace=True)
        test = pd.read_csv(f'{self.data_dir}/annotations/test.csv')
        return test.merge(annotations, on='video_name', how='left').reset_index(drop=True)

    def heatmaps(self, df):
        heatmap_df = []
        print('making the heatmaps...')
        missing_trials = 0
        total_trials = 0
        for (video, block, run), rep_df in df.groupby(['video_name', 'run', 'block']):
            single_heatmap = generate_heatmap(rep_df.x.to_numpy(), rep_df.y.to_numpy(),
                                              scaling_factor=(self.presentation_size / self.res),
                                              x_max=self.screen_width, border=self.x_border,
                                              y_max=self.presentation_size, n_samples=self.n_samples)
            total_trials += 1
            if np.all(np.isnan(single_heatmap)):
                missing_trials += 1
            heatmap_df.append({'video_name': video, 'block': block, 'run': run,
                               'heatmap': single_heatmap.flatten()})
        heatmap_df = pd.DataFrame(heatmap_df)
        heatmap_df = pd.concat([heatmap_df, pd.DataFrame(heatmap_df["heatmap"].tolist()).add_prefix("heatmap")], axis=1)
        heatmap_df.drop(columns=['heatmap'], inplace=True)
        heatmap_df['repetition'] = heatmap_df.groupby(['video_name']).cumcount()
        heatmap_df['even'] = heatmap_df.repetition % 2
        heatmap_df['even'] = heatmap_df.even.astype('bool')
        print(f'{missing_trials} missing trials out of {total_trials} total trials')
        print(f'{(missing_trials/total_trials)*100:.2f} % missing trials')
        return heatmap_df

    def heatmap_visualization(self, average_heatmap, bb):
        import cv2, mmcv
        def load_frame(path, video, frame_num=0):
            in_file = f'{path}/videos/{video}'
            video_obj = mmcv.VideoReader(in_file)
            frame = video_obj[frame_num]
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.putalpha(255)
            return frame

        heatmap_cols = [f'heatmap{i}' for i in range(self.res**2)]
        print('saving heatmap visualization...')
        for _, row in tqdm(average_heatmap.iterrows(), total=len(average_heatmap)):
            heatmap = row[heatmap_cols].to_numpy().astype('float').reshape((self.res, self.res))
            if not np.all(np.isnan(heatmap)):
                frame = load_frame(self.data_dir, row.video_name)
                file = f"{self.figure_dir}/heatmaps/subj{self.sid}/{row.video_name.replace('mp4', 'png')}"
                heatmap_img = heatmap2img(heatmap, frame.size)

                if (row.video_name, 1, 'face1') in bb.index:
                    bb1 = str2ints(bb.loc[row.video_name, 1, 'face1'].box)
                else:
                    bb1 = None

                if (row.video_name, 1, 'face2') in bb.index:
                    bb2 = str2ints(bb.loc[row.video_name, 1, 'face2'].box)
                else:
                    bb2 = None

                plt_fixation(frame, heatmap_img, bb1, bb2, file)
            else:
                print(f'no data for {row.video_name}')

    def plot_histogram(self, df_):
        _, ax = plt.subplots()
        ax.hist(df_.r)
        ax.vlines(x=df_.mean(numeric_only=True).r, ymin=0, ymax=10, colors='r')
        plt.savefig(f'{self.figure_dir}/intrasubj_reliability/subj{self.sid}.pdf')
        plt.close('all')

    def intrasubj_reliability(self, split_half):
        columns = [col for col in split_half.columns if 'heatmap' in col]
        out = []
        for vid, vid_df in split_half.groupby('video_name'):
            even = vid_df.loc[vid_df.even == 1, columns].to_numpy().astype('float')[0]
            odd = vid_df.loc[vid_df.even == 0, columns].to_numpy().astype('float')[0]
            if np.all(np.isnan(even)) or np.all(np.isnan(odd)):
                r = np.nan
            else:
                r = corr(even, odd)
            out.append({'video_name': vid, 'r': r})
        out = pd.DataFrame(out)
        print(f'intrasubj reliability = {np.nanmean(out.r.to_numpy())}')
        out.to_csv(f'{self.out_dir}/{self.process}/intrasubj_reliability/subj{self.sid}.csv', index=False)
        self.plot_histogram(out)
        return out

    def save_heatmaps_csv(self, out_):
        out_.to_csv(f'{self.out_dir}/{self.process}/heatmaps/subj{self.sid}.csv', index=False)

    def plot_results(self, df, out_name):
        sns.barplot(x='feature', y='r', data=df, color='gray')
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.savefig(out_name)
        plt.close('all')

    def analysis_annot_corr(self, main_data, main_col, analysis,
                                 annotations, features):
        print(f'starting on {analysis} analysis')
        merged = main_data.merge(annotations, on='video_name', how='left')
        out_df = []
        for feature in features:
            print(feature)
            r, p, r_null = perm(merged[feature].to_numpy(), merged[main_col].to_numpy(),
                                H0='two_tailed', square=False)
            r_var = bootstrap(merged[feature].to_numpy(), merged[main_col].to_numpy(),
                               square=False)
            out_df.append({'feature': feature, 'r': r, 'p': p,
                           'r_null': r_null, 'r_var': r_var})
        out_df = pd.DataFrame(out_df)
        self.plot_results(out_df, f'{self.figure_dir}/{analysis}/subj{self.sid}.pdf')
        out_df.to_pickle(f'{self.out_dir}/{self.process}/{analysis}/subj{self.sid}.pkl')

    def run(self):
        df = self.load_data()
        annotations = self.load_ratings()
        if not os.path.exists(f'{self.out_dir}/{self.process}/heatmaps/subj{self.sid}.csv') or self.overwrite:
            # bounding_boxes, bb_map = load_bounding_boxes(self.data_dir, self.frame_size, self.res)
            heatmap_df = self.heatmaps(df)
            # heatmap_df = face_looking(heatmap_df, bb_map, self.res)
            # heatmap_df = right_looking(heatmap_df, self.res)
            # self.save_heatmaps_csv(heatmap_df)
        else:
            heatmap_df = pd.read_csv(f'{self.out_dir}/{self.process}/heatmaps/subj{self.sid}.csv')

        if self.save_heatmaps:
            bounding_boxes, _ = load_bounding_boxes(self.data_dir, self.frame_size, self.res)
            self.heatmap_visualization(heatmap_df.groupby(['video_name']).mean(numeric_only=True).reset_index(drop=False),
                                       bounding_boxes)

        # reliability = self.intrasubj_reliability(heatmap_df.groupby(['video_name', 'even']).mean(numeric_only=True).reset_index(drop=False))
        # self.analysis_annot_corr(reliability, 'r', 'reliability_feature',
        #                          annotations, self.features)
        # self.analysis_annot_corr(heatmap_df, 'prop_face_looking', 'face_feature',
        #                          annotations, self.features)
        # self.analysis_annot_corr(heatmap_df, 'prop_right_looking', 'right_feature',
        #                          annotations, ['transitivity', 'communication'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default='15')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_heatmaps', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--resolution', type=int, default=20)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    EyeTracking_WithinSubj(args).run()


if __name__ == '__main__':
    main()
