#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
import tools
from tqdm import tqdm

def inner_ridge(X_train_, y_train_, X_test_, n_features, n_splits=4): 
    #Standardize X
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)

    #Ridge CV to get the alpha parameter
    clf = RidgeCV(cv=n_splits).fit(X_train_, y_train_)

    #Fit the Ridge model with the found best alpha
    lr = Ridge(fit_intercept=False, alpha=clf.alpha_).fit(X_train_, y_train_)

    #Scale testing 
    mean = scaler.mean_[:X_test_.shape[1]]
    var = scaler.var_[:X_test_.shape[1]]
    X_test_ = (X_test_ - mean)/var

    # If y is 2D, there should be a prediction for each voxel
    if len(y_train_.shape) > 1:
        y_pred = np.zeros((n_features, X_test_.shape[0], y_train_.shape[-1]))
    else:
        y_pred = np.zeros((n_features, X_test_.shape[0]))
    
    for ifeature in range(n_features):
        if len(y_train_.shape) > 1:
            for i in range(X_test_.shape[0]):
                y_pred[ifeature, i, :] = np.multiply(X_test_[i, ifeature], lr.coef_[:, ifeature])
        else:
            y_pred[ifeature, :] = X_test_[:, ifeature] * lr.coef_[ifeature]
    return y_pred

def outer_ridge_2d(X, X_control, y, n_features, splitter,
                  n_conditions=200, mask_path=None):
    # Mask the brain
    tot_voxels = y.shape[0]
    y, y_inds = tools.mask(y, mask_path)
    
    # Get counts
    y = y.T
    n_voxels = y.shape[-1]
    n_splits = splitter.n_splits
    n_condpersplit = int(n_conditions/n_splits)

    # Iterate through the different splits
    y_pred = np.zeros((n_splits, n_features, n_condpersplit, n_voxels))
    y_true = np.zeros((n_splits, n_condpersplit, n_voxels))
    test_inds = np.zeros((n_splits, n_condpersplit), dtype='int')
    for i, (train_index, test_index) in tqdm(enumerate(splitter.split(X)), total=n_splits):
        # Split the training and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # save the current indices
        test_inds[i, ...] = test_index
        y_true[i, ...] = y_test

        # new X - combine the annotated features with the nuissance regressors
        X_train = np.append(X_train, X_control[train_index], axis=1)

        # Prediction
        y_pred[i, ...] = inner_ridge(X_train, y_train, X_test, n_features)
    y_pred = np.swapaxes(y_pred, 0, 1).reshape((n_features, n_conditions, -1))
    y_true = y_true.reshape((n_conditions, n_voxels))
    return y_true, y_pred, test_inds