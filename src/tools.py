#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
import nibabel as nib


def filter_r(rs, ps, p_crit=0.05, correct=True, threshold=True):
    if correct and (ps is not None):
        _, ps_corrected, _, _ = multipletests(ps, method='fdr_bh')
    elif not correct and (ps is not None):
        ps_corrected = ps.copy()
    else:
        ps_corrected = None

    if threshold and (ps is not None):
        rs[ps_corrected >= p_crit] = 0.
    else:
        rs[rs < 0.] = 0.
    return rs, ps_corrected


def corr2d(x, y):
    x_m = x - x.mean(axis=0)
    y_m = y - y.mean(axis=0)

    numer = np.sum((x_m * y_m), axis=0)
    denom = np.sqrt(np.sum((x_m * x_m), axis=0) * np.sum((y_m * y_m), axis=0))
    denom[denom == 0] = np.NaN
    return numer / denom


def mantel_permutation(a, i):
    a = squareform(a)
    inds = np.random.permutation(a.shape[0])
    a_shuffle = a[inds][:, inds]
    return squareform(a_shuffle)


def calculate_p(r_null_, r_true_, n_perm_, H0_):
    # Get the p-value depending on the type of test
    denominator = n_perm_ + 1
    if H0_ == 'two_tailed':
        numerator = np.sum(np.abs(r_null_) >= np.abs(r_true_), axis=0) + 1
        p_ = numerator / denominator
    elif H0_ == 'greater':
        numerator = np.sum(r_true_ > r_null_, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    else:  # H0 == 'less':
        numerator = np.sum(r_true_ < r_null_, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    return p_


def bootstrap(a, b, n_perm=int(5e3)):
    if a.ndim == 3:
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])

    r2_var = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                               size=a.shape[0])
        if a.ndim == 3:
            a_sample = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        else:  # a.ndim == 2:
            a_sample = a[inds, :]
        r2_var[i, :] = corr2d(a_sample, b)**2

    return r2_var


def bootstrap_unique_variance(a, b, c, n_perm=int(5e3)):
    if a.ndim == 3:
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
        c = c.reshape(c.shape[0] * c.shape[1], c.shape[-1])

    # Shuffle a and recompute r^2 n_perm times
    r2_var = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                               size=a.shape[0])
        if a.ndim == 3:
            a_sample = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        else:  # a.ndim == 2:
            a_sample = a[inds, :]
        r2_var[i, :] = corr2d(a_sample, b)**2 - corr2d(a_sample, c)**2
    return r2_var


def perm(a, b, n_perm=int(5e3), H0='greater'):
    if a.ndim == 3:
        a_not_shuffle = a.reshape(a.shape[0] * a.shape[1], a.shape[-1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
        r2 = corr2d(a_not_shuffle, b)**2
    else:
        r2 = corr2d(a, b)**2

    r2_null = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).permutation(a.shape[0])
        if a.ndim == 3:
            a_shuffle = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        else:  # a.ndim == 2:
            a_shuffle = a[inds, :]
        r2_null[i, :] = corr2d(a_shuffle, b)**2

    # Get the p-value depending on the type of test
    p = calculate_p(r2_null, r2, n_perm, H0)
    return r2, p, r2_null


def perm_unique_variance(a, b, c, n_perm=int(5e3), H0='greater'):
    if a.ndim == 3:
        a_not_shuffle = a.reshape(a.shape[0] * a.shape[1], a.shape[-1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
        c = c.reshape(c.shape[0] * c.shape[1], c.shape[-1])
        r2 = corr2d(a_not_shuffle, b)**2 - corr2d(a_not_shuffle, c)**2
    else:
        r2 = corr2d(a, b)**2 - corr2d(a, c)**2

    # Shuffle a and recompute r^2 n_perm times
    r2_null = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).permutation(a.shape[0])
        if a.ndim == 3:
            a_shuffle = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        else:  # a.ndim == 2:
            a_shuffle = a[inds, :]
        r2_null[i, :] = corr2d(a_shuffle, b)**2 - corr2d(a_shuffle, c)**2

    p = calculate_p(r2_null, r2, n_perm, H0)
    return r2, p, r2_null


def compute_confidence_interval(distribution):
    return np.percentile(distribution, [2.5, 97.5])


def mask_img(img, mask, fill=0.):
    if type(img) is nib.nifti1.Nifti1Image:
        masked_img = np.array(img.dataobj)
        mask = np.array(mask.dataobj)
    else:
        masked_img = img.copy()
    mask = np.invert(mask.astype('bool'))
    i, j, k = np.where(mask)
    masked_img[i, j, k] = fill
    if type(img) is nib.nifti1.Nifti1Image:
        masked_img = nib.Nifti1Image(masked_img, img.affine, img.header)
    return masked_img
