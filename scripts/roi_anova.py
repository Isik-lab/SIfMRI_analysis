#!/usr/bin/env python
# coding: utf-8

import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from src import tools
import pickle


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


def subtract_rois(data, roi1, roi2):
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


out_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim'
data_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw'
prior_process = 'VoxelPermutation'
process = 'ROI_ANOVA'
Path(f'{out_dir}/{process}').mkdir(exist_ok=True, parents=True)
cat1 = 'social'
cat2 = 'scene_object'
roi1 = 'pSTS'
roi2 = 'FFA'
for sid in ['01', '02', '03', '04']:
    reliability_file = f'{out_dir}/Reliability/sub-{sid}_space-T1w_desc-test-fracridge_reliability-mask.nii.gz'
    r2_cats = subtract_cats(f'{out_dir}/{prior_process}/sub-{sid}_dropped-categorywithnuisance',
                            cat1, cat2, 'r2.nii.gz')
    r2null_cats = subtract_cats(f'{out_dir}/{prior_process}/dist/sub-{sid}_dropped-categorywithnuisance',
                                cat1, cat2, 'r2null.npy')
    r2var_cats = subtract_cats(f'{out_dir}/{prior_process}/dist/sub-{sid}_dropped-categorywithnuisance',
                               cat1, cat2, 'r2var.npy')

    data = {'sid': sid, 'roi1': roi1, 'roi2': roi2, 'cat1': cat1, 'cat2': cat2,
            'r2': subtract_rois(r2_cats, roi1, roi2)}
    r2null_rois = subtract_rois(r2null_cats, roi1, roi2)
    r2var_rois = subtract_rois(r2var_cats, roi1, roi2)
    data['p'] = compute_significance(data['r2'], r2null_rois)
    data['low_ci'], data['high_ci'] = np.percentile(r2var_rois, [2.5, 97.5])
    print(data)
    save_results(data,
                 f'{out_dir}/{process}/sub-{sid}_rois-{roi1}-{roi2}_categories-{cat1}-{cat2}.pkl')
