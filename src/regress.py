#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.decomposition import PCA
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
    return lr, lr.coef_


def scale_by_feature(X_train_, X_test_, n_features):
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)
    mean = scaler.mean_[:n_features]
    var = scaler.var_[:n_features]
    return X_train_, (X_test_ - mean) / var


def scale(X_train_, X_test_):
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)
    if X_test_ is not None:
        scaler.transform(X_test_)
    return X_train_, X_test_


def predict_by_feature(X_test_, y_train_, betas, n_features):
    # Make the prediction for each voxel
    y_pred = np.zeros((n_features, X_test_.shape[0], y_train_.shape[-1]))
    for i_feature in range(n_features):
        for i in range(X_test_.shape[0]):
            y_pred[i_feature, i, :] = np.multiply(X_test_[i, i_feature],
                                                  betas[:, i_feature])
    return y_pred


def predict(model, X_test_):
    if X_test_ is not None:
        return model.predict(X_test_)
    else:
        return None

def pca(X_train_, X_test_):
    pca_ = PCA(svd_solver='full', whiten=True)
    X_train_ = pca_.fit_transform(X_train_)
    return X_train_, pca_.transform(X_test_)

def cross_validated_ridge(X, X_control,
                          y, n_features,
                          splitter,
                          by_feature=False,
                          include_control=False,
                          n_conditions=200):
    # Get counts
    y = y.T
    n_voxels = y.shape[-1]
    n_splits = splitter.n_splits
    n_conds_per_split = int(n_conditions / n_splits)

    # Iterate through the different splits
    if by_feature:
        y_pred = np.zeros((n_splits, n_features, n_conds_per_split, n_voxels))
    else:
        y_pred = np.zeros((n_splits, n_conds_per_split, n_voxels))
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
        if include_control:
            X_train = np.hstack((X_train, X_control[train_index]))
            if not by_feature:
                X_test = np.hstack((X_test, X_control[test_index]))

        # Standardize X
        if by_feature:
            X_train, X_test = scale_by_feature(X_train, X_test, n_features)
        else:
            X_train, X_test = scale(X_train, X_test)

        # Orthogonalize
        if not by_feature:
            X_train, X_test = pca(X_train, X_test)

        # Find alpha
        alpha = inner_ridge(X_train, y_train)

        # Fit the regression
        model, betas = outer_ridge(X_train, y_train, alpha)

        # Prediction
        if by_feature:
            y_pred[i, ...] = predict_by_feature(X_test, y_train, betas, n_features)
        else:
            y_pred[i, ...] = predict(model, X_test)

    if by_feature:
        y_pred = np.swapaxes(y_pred, 0, 1).reshape((n_features, n_conditions, n_voxels))
    else:
        y_pred = y_pred.reshape((n_conditions, n_voxels))

    y_true = y_true.reshape((n_conditions, n_voxels))
    return y_true, y_pred, test_indices


def ridge(X_train, y_train, X_test=None):
    # Standardize X
    X_train, X_test = scale(X_train, X_test)

    # Find alpha
    alpha = inner_ridge(X_train, y_train)

    # Fit the regression
    model, betas = outer_ridge(X_train, y_train, alpha)

    # Prediction
    return model.coef_, predict(model, X_test)
