#!/usr/bin/env python
# coding: utf-8

import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

from src import tools
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools



def subtract_cats(prefix, cat1, cat2, suffix):
    if 'npy' in suffix:
        arr1 = np.load(f'{prefix}-{cat1}_{suffix}')
        arr2 = np.load(f'{prefix}-{cat2}_{suffix}')
    else:
        arr1 = np.array(nib.load(f'{prefix}-{cat1}_{suffix}').dataobj)
        arr2 = np.array(nib.load(f'{prefix}-{cat2}_{suffix}').dataobj)
    return arr1 - arr2


def mask_img(img, mask):
    if type(img) is str:
        img = nib.load(img)
        img = np.array(img.dataobj)

    if type(mask) is str:
        mask = nib.load(mask)

    mask = np.array(mask.dataobj, dtype=bool)
    return img[mask].squeeze()


def load_roi_hemi_mask(data_dir, sid, roi, hemi, reliability_file):
    roi_file = glob.glob(f'{data_dir}/localizers/sub-{sid}/sub-{sid}*{roi}*{hemi}*mask.nii.gz')[
        0]
    roi_mask = mask_img(roi_file, reliability_file).astype('bool')
    return roi_mask


def extract_roi_activity(data, roi_mask, reliability_file):
    if data.ndim == 3:
        reliable_data = mask_img(data, reliability_file)
        roi_data = reliable_data[roi_mask]
    else:
        roi_data = data[:, roi_mask]
    return roi_data


def concat_rois(data, sid, roi, reliability_file):
    roi_activity = []
    for hemi in ['lh', 'rh']:
        roi_mask = load_roi_hemi_mask(data_dir, sid, roi, hemi, reliability_file)
        roi_activity.append(extract_roi_activity(data, roi_mask, reliability_file).T)
    return np.concatenate(roi_activity).mean(axis=0)


def subtract_rois(data, roi1, roi2, sid, reliability_file):
    data_roi1 = concat_rois(data, sid, roi1, reliability_file)
    data_roi2 = concat_rois(data, sid, roi2, reliability_file)
    return data_roi1 - data_roi2


def compute_significance(r2, r2null):
    return tools.calculate_p(r2null, r2,
                             n_perm_=len(r2null),
                             H0_='greater')


def save_results(d, out_file_name):
    f = open(out_file_name, "wb")
    pickle.dump(d, f)
    f.close()


def load_pkl(file):
    d = pickle.load(open(file, 'rb'))
    return d


def compute_anova(out_dir, prior_process, process, rois, roi2):
    if not os.path.exists(f'{out_dir}/{process}/sub-01_rois-aSTS-FFA_categories-social-scene_object.pkl'):
        for roi1 in rois:
            for sid in ['01', '02', '03', '04']:
                reliability_file = f'{out_dir}/Reliability/sub-{sid}_space-T1w_desc-test-fracridge_reliability-mask.nii.gz'
                r2_cats = subtract_cats(f'{out_dir}/{prior_process}/sub-{sid}_dropped-categorywithnuisance',
                                        cat1, cat2, 'r2.nii.gz')
                r2null_cats = subtract_cats(f'{out_dir}/{prior_process}/dist/sub-{sid}_dropped-categorywithnuisance',
                                            cat1, cat2, 'r2null.npy')
                r2var_cats = subtract_cats(f'{out_dir}/{prior_process}/dist/sub-{sid}_dropped-categorywithnuisance',
                                           cat1, cat2, 'r2var.npy')

                data = {'sid': sid, 'roi1': roi1, 'roi2': roi2, 'cat1': cat1, 'cat2': cat2,
                        'r2': subtract_rois(r2_cats, roi1, roi2, sid, reliability_file)}
                r2null_rois = subtract_rois(r2null_cats, roi1, roi2, sid, reliability_file)
                r2var_rois = subtract_rois(r2var_cats, roi1, roi2, sid, reliability_file)
                data['p'] = compute_significance(data['r2'], r2null_rois)
                data['low_ci'], data['high_ci'] = np.percentile(r2var_rois, [2.5, 97.5])
                print(data)
                save_results(data,
                             f'{out_dir}/{process}/sub-{sid}_rois-{roi1}-{roi2}_categories-{cat1}-{cat2}.pkl')
    else:
        data = []
        for file in glob.glob(f'{out_dir}/{process}/*.pkl'):
            in_data = load_pkl(file)
            data.append(in_data)
        df = pd.DataFrame(data)

        df.sid = pd.Categorical(df.sid, categories=subjs, ordered=True)
        df.roi1 = pd.Categorical(df.roi1, categories=rois, ordered=True)
        df['significant'] = 'ns'
        for i, row in df.iterrows():
            if 0.05 > row.p > 0.01:
                df.at[i, 'significant'] = '*'
            elif 0.01 >= row.p > 0.001:
                df.at[i, 'significant'] = '**'
            elif 0.001 >= row.p:
                df.at[i, 'significant'] = '***'
        df.to_csv(f'{out_dir}/{process}/anova.csv', index=False)
    return df


def subj2shade(key):
    d = {'01': 1.0,
         '02': 0.8,
         '03': 0.6,
         '04': 0.4}
    return d[key]


def plot_results(figure_dir, df, rois, subjs):
    sns.set_theme(font_scale=2.5)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='poster', style='white', rc=custom_params)
    _, ax = plt.subplots(1, figsize=(8, 8))

    sns.barplot(x='roi1', y='r2',
                hue='sid', color=[0.8, 0, 0.8],
                ax=ax, data=df)

    y_max = df.high_ci.max() + 0.03
    ax.set_ylim([0, y_max])

    ticks = [f'{roi}-SI - FFA' for roi in rois]
    ax.set_xticklabels(ticks, fontsize=22)

    # Plot vertical lines to separate the bars
    for x in np.arange(0.5, len(rois) - 0.5):
        ax.plot([x, x], [0, y_max - (y_max / 20)], '0.8')

    # Manipulate the color and add error bars
    for bar, (subj, roi) in zip(ax.patches, itertools.product(subjs, rois)):
        color = np.array(bar.get_facecolor())
        color[:-1] = color[:-1] * subj2shade(subj)
        y1 = df.loc[(df.sid == subj) & (df.roi1 == roi),
                    'low_ci'].item()
        y2 = df.loc[(df.sid == subj) & (df.roi1 == roi),
                    'high_ci'].item()
        sig = df.loc[(df.sid == subj) & (df.roi1 == roi),
                     'significant'].item()
        x = bar.get_x() + 0.1
        ax.plot([x, x], [y1, y2], 'k')
        if sig != 'ns':
            # ax.scatter(x, y_max - 0.01, marker='o', color=color, edgecolors=[0.2, 0.2, 0.2])
            ax.text(x, y_max - 0.02, sig,
                    horizontalalignment='center')
        bar.set_facecolor(color)
        bar.set_edgecolor((.2, .2, .2))
    ax.legend([], [], frameon=False)
    ax.set_ylabel('SI - SO unique variance ($r^2$)', fontsize=26)
    ax.set_xlabel('ROI difference', fontsize=26)
    plt.tight_layout()
    plt.savefig(f'{figure_dir}/anova.pdf')

prior_process = 'VoxelPermutation'
process = 'ROI_ANOVA'
top_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis'
out_dir = f'{top_dir}/data/interim'
data_dir = f'{top_dir}/data/raw'
figure_dir = f'{top_dir}/reports/figures/{process}'
Path(f'{out_dir}/{process}').mkdir(exist_ok=True, parents=True)
Path(figure_dir).mkdir(exist_ok=True, parents=True)
cat1 = 'social'
cat2 = 'scene_object'
rois = ['pSTS', 'aSTS']
roi2 = 'FFA'
subjs = ['01', '02', '03', '04']

df = compute_anova(out_dir, prior_process, process, rois, roi2)
plot_results(figure_dir, df, rois, subjs)

