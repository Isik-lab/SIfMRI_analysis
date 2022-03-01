#!/usr/bin/env python
# coding: utf-8

import numpy as np

def corr2d(x, y):
    x_m = x - x.mean(axis=0)
    y_m = y - y.mean(axis=0)

    numer = np.sum((x_m * y_m), axis=0)
    denom = np.sqrt(np.sum((x_m * x_m), axis=0) * np.sum((y_m * y_m), axis=0))
    return numer / denom

def corr1d(x, y):
    """
        input:
            x:
            y:
        output:
    """
    x_m = x - x.mean()
    y_m = y - y.mean()

    num = x_m @ y_m
    denom = np.sqrt((x_m @ x_m) * (y_m @ y_m))
    if denom != 0 :
        return num / denom
    else:
        return np.NaN

def permutation_test(a, b, test_inds=None,
                     n_perm=int(5e3), H0='greater'):
    r_true = corr1d(a, b)
    r_null = np.zeros(n_perm)
    for i in range(n_perm):
        inds = np.random.default_rng(i).permutation(test_inds.shape[0])
        inds = test_inds[inds, :].flatten()
        a_shuffle = a[inds]
        r_null[i] = corr1d(a_shuffle, b)

    #Get the p-value depending on the type of test
    if H0 == 'two_tailed':
        p = np.sum(np.abs(r_null) >= np.abs(r_true)) / n_perm
    elif H0 == 'greater':
        p = 1 - (np.sum(r_true >= r_null) / n_perm)
    elif H0 == 'less':
        p = 1 - (np.sum(r_true <= r_null) / n_perm)

    return r_true, p, r_null

def permutation_test2d(a, b, test_inds=None,
                     n_perm=int(5e3), H0='greater'):
    r_true = corr2d(a, b)
    r_null = np.zeros((n_perm, a.shape[-1]))
    for i in range(n_perm):
        print(f'Starting permutation {i}')
        inds = np.random.default_rng(i).permutation(test_inds.shape[0])
        inds = test_inds[inds, :].flatten()
        a_shuffle = a[inds, :]
        r_null[i, :] = corr2d(a_shuffle, b)

    #Get the p-value depending on the type of test
    if H0 == 'two_tailed':
        p = np.sum(np.abs(r_null) >= np.abs(r_true), axis=0) / n_perm
    elif H0 == 'greater':
        p = 1 - (np.sum(r_true >= r_null, axis=0) / n_perm)
    elif H0 == 'less':
        p = 1 - (np.sum(r_true <= r_null, axis=0) / n_perm)

    p[np.isnan(r_true)] = np.NaN
    r_null[:, np.isnan(r_true)] = np.NaN
    return r_true, p, r_null

def bootstrap(a, b, test_inds, n_samples=int(5e3)):
    r_var = np.zeros(n_samples)
    for i in range(n_samples):
        inds = np.random.default_rng(i).choice(np.arange(test_inds.shape[0]),
                                               size=test_inds.shape[0])
        inds = test_inds[inds, :].flatten()
        r_var[i] = corr1d(a[inds], b)
    return r_var

def mask(stat_map, path=None):
    #activity in ROI
    if path is not None:
        from nilearn import image
        m = image.load_img(path)
        m = np.array(m.dataobj, dtype='bool').flatten()
    else:
        m = np.ones(stat_map.shape[0], dtype='bool')
    roi_activation = stat_map[m, ...]

    #Remove nan values (these are voxels that do not vary across the different videos)
    if len(stat_map.shape) == 2:
        inds = ~np.any(np.isnan(roi_activation), axis=1)
    else:
        inds = np.ones(roi_activation.shape[0], dtype='bool')
    return roi_activation[inds, ...], inds
