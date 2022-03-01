#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from tqdm import tqdm


def inner_ridge(X_train_, y_train_, n_splits=4):
    # Within each loop, find clf.alpha_ with RidgeCV
    clf = RidgeCV(cv=n_splits, scoring="r2")
    clf.fit(X_train_, y_train_)
    return clf.alpha_


def outer_ridge(X_train_, y_train_, alpha):
    # Fit the Ridge model with the selected alpha
    lr = Ridge(fit_intercept=False, alpha=alpha)
    lr.fit(X_train_, y_train_)
    return lr.coef_


def scale(X_train_, X_test_, n_features):
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)
    mean = scaler.mean_[:n_features]
    var = scaler.var_[:n_features]
    return X_train_, (X_test_ - mean) / var


def predict(X_test_, y_train_, betas, n_features):
    # Make the prediction for each voxel
    y_pred = np.zeros((n_features, X_test_.shape[0], y_train_.shape[-1]))
    for i_feature in range(n_features):
        for i in range(X_test_.shape[0]):
            y_pred[i_feature, i, :] = np.multiply(X_test_[i, i_feature],
                                                  betas[:, i_feature])
    return y_pred


def regress(X_train_, X_test_, y_train_, n_features):
    # Standardize X
    X_train_, X_test_ = scale(X_train_, X_test_, n_features)

    # Find alpha
    alpha = inner_ridge(X_train_, y_train_)

    # Fit the regression
    betas = outer_ridge(X_train_, y_train_, alpha)

    # Predict
    return predict(X_test_, y_train_, betas, n_features)


def cross_validated_ridge(X, X_control, y, n_features, splitter,
                          n_conditions=200):
    # Get counts
    y = y.T
    n_voxels = y.shape[-1]
    n_splits = splitter.n_splits
    n_conds_per_split = int(n_conditions / n_splits)

    # Iterate through the different splits
    y_pred = np.zeros((n_splits, n_features, n_conds_per_split, n_voxels))
    y_true = np.zeros((n_splits, n_conds_per_split, n_voxels))
    test_indices = np.zeros((n_splits, n_conds_per_split), dtype='int')
    for i, (train_index, test_index) in tqdm(enumerate(splitter.split(X)), total=n_splits):
        # Split the training and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # save the current indices
        test_indices[i, ...] = test_index
        y_true[i, ...] = y_test

        # new X - combine the annotated features with the nuisance regressors
        X_train = np.hstack((X_train, X_control[train_index]))

        # Prediction
        y_pred[i, ...] = regress(X_train, X_test, y_train, n_features)
    y_pred = np.swapaxes(y_pred, 0, 1).reshape((n_features, n_conditions, n_voxels))
    y_true = y_true.reshape((n_conditions, n_voxels))
    return y_true, y_pred, test_indices
