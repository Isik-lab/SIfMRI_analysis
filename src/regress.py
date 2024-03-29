#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from tqdm import tqdm


def inner_ridge(X_train_, y_train_, n_splits=4,
                random_state=0):
    # Within each loop, find clf.alpha_ with RidgeCV
    alphas = 10. ** np.arange(start=-1., stop=6.)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = RidgeCV(cv=kf, scoring="r2", alphas=alphas)
    clf.fit(X_train_, y_train_)
    return clf.alpha_


def outer_ridge(X_train_, y_train_, alpha):
    # Fit the Ridge model with the selected alpha
    lr = Ridge(fit_intercept=False, alpha=alpha)
    lr.fit(X_train_, y_train_)
    return lr, lr.coef_


def scale(train_, test_):
    n_features = test_.shape[-1]
    scaler = StandardScaler()
    train_ = scaler.fit_transform(train_)
    mean = scaler.mean_[:n_features].squeeze()
    var = scaler.var_[:n_features].squeeze()
    var[np.isclose(var, 0.)] = np.nan
    return train_, (test_ - mean) / var


def get_feature_inds(filter_features):
    features = ['indoor', 'expanse', 'transitivity', 'agent distance',
                'facingness', 'joint action', 'communication', 'cooperation', 'dominance',
                'intimacy', 'valence', 'arousal']

    if type(filter_features) is str:
        filter_features = [filter_features]

    inds = []
    for f in filter_features:
        ind = features.index(f)
        print(ind)
        inds.append(ind)
    return inds


def predict_multi_feature(X_test_, y_train_, betas, inds):
    n_features = len(inds)
    if len(betas.shape) > 1:
        # Make the prediction for each voxel
        y_pred = np.zeros((n_features, X_test_.shape[0], y_train_.shape[-1]))
        out_betas = np.zeros((y_train_.shape[-1], n_features))
        for count, i_feature in enumerate(inds):
            y_pred[count, :, :] = X_test_[:, i_feature:i_feature+1] @ betas[:, i_feature:i_feature+1].T
            out_betas[:, count] = betas[:, i_feature]
        # add up the prediction for each of the features as would be done in the matrix multiplication
        y_pred = y_pred.sum(axis=0)
    else: #1D prediction, like for the ROI analysis
        out_betas = None
        y_pred = np.zeros((n_features, X_test_.shape[0]))
        for count, i_feature in enumerate(inds):
            y_pred[count, :] = np.multiply(X_test_[:, i_feature], betas[i_feature])
        # add up the prediction for each of the features as would be done in the matrix multiplication
        y_pred = y_pred.sum(axis=0)
    return y_pred, out_betas


def predict(X_test_, y_train_, betas, n_nuisance=40, by_feature=False):
    n_features = betas.shape[-1] - n_nuisance
    if len(betas.shape) > 1:
        # Make the prediction for each voxel
        out_betas = np.zeros((y_train_.shape[-1], n_features))
        y_pred = np.zeros((n_features, X_test_.shape[0], y_train_.shape[-1]))
        for i_feature in range(n_features):
            y_pred[i_feature, :, :] = X_test_[:, i_feature:i_feature+1] @ betas[:, i_feature:i_feature+1].T
            out_betas[:, i_feature] = betas[:, i_feature]
        if not by_feature:
            # add up the prediction for each of the features as would be done in the matrix multiplication
            y_pred = y_pred.sum(axis=0)
    else: #1D prediction, like for the ROI analysis
        out_betas = None
        y_pred = np.zeros((n_features, X_test_.shape[0]))
        for i_feature in range(n_features):
            y_pred[i_feature, :] = np.multiply(X_test_[:, i_feature], betas[i_feature])
        if not by_feature:
            # add up the prediction for each of the features as would be done in the matrix multiplication
            y_pred = y_pred.sum(axis=0)
    return y_pred, out_betas


def pca(X_train_, X_test_):
    pca_ = PCA(svd_solver='full', whiten=True)
    X_train_ = pca_.fit_transform(X_train_)
    return X_train_, pca_.transform(X_test_)


def cross_validated_ridge(X, X_control,
                          y, n_features,
                          splitter,
                          predict_by_feature=False,
                          include_control=False,
                          pca_before_regression=False,
                          n_conditions=200,
                          inds=None):
    # Get counts
    if len(y.shape) > 1:
        y = y.T
        n_voxels = y.shape[-1]
    else:
        n_voxels = 1
    n_splits = splitter.n_splits
    n_conds_per_split = int(n_conditions / n_splits)

    if inds is not None:
        n_features = len(inds)

    # Iterate through the different splits
    if predict_by_feature:
        y_pred = np.zeros((n_splits, n_features, n_conds_per_split, n_voxels)).squeeze()
    else:
        y_pred = np.zeros((n_splits, n_conds_per_split, n_voxels)).squeeze()
    betas = np.zeros((n_splits, n_voxels, n_features))
    y_true = np.zeros((n_splits, n_conds_per_split, n_voxels)).squeeze()
    test_indices = np.zeros((n_splits, n_conds_per_split), dtype='int')
    for i, (train_index, test_index) in tqdm(enumerate(splitter.split(X)), total=n_splits):
        # Split the training and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # new X - combine the annotated features with the nuisance regressors
        if include_control:
            n_nuissance = X_control.shape[-1]
            X_train = np.hstack((X_train, X_control[train_index]))
        else:
            n_nuissance = 0

        # Standardize data
        X_train, X_test = scale(X_train, X_test)
        y_train, y_test = scale(y_train, y_test)

        # save the current indices
        test_indices[i, ...] = test_index
        y_true[i, ...] = y_test

        # Orthogonalize
        if pca_before_regression:
            X_train, X_test = pca(X_train, X_test)

        # Find alpha
        alpha = inner_ridge(X_train, y_train)

        # Fit the regression
        model, lr_coef = outer_ridge(X_train, y_train, alpha)

        # Prediction
        if inds is None:
            if predict_by_feature:
                y_pred[i, ...], out_betas = predict(X_test, y_train, lr_coef, n_nuissance, by_feature=True)
            else:
                y_pred[i, ...], out_betas = predict(X_test, y_train, lr_coef, n_nuissance)
        else:
            y_pred[i, ...], out_betas = predict_multi_feature(X_test, y_train, lr_coef, inds)

        if out_betas is not None:
            betas[i, ...] = out_betas

    if predict_by_feature:
        y_pred = np.swapaxes(y_pred, 0, 1).reshape((n_features, n_conditions, n_voxels)).squeeze()
    else:
        y_pred = y_pred.reshape((n_conditions, n_voxels)).squeeze()

    y_true = y_true.reshape((n_conditions, n_voxels)).squeeze()
    return y_true, y_pred, test_indices, betas.mean(axis=0)


def ridge(X_train, X_control, X_test,
          y_train, y_test,
          include_control=False,
          predict_by_feature=False,
          inds=None):

    if include_control:
        n_nuissance = X_control.shape[-1]
        X_train = np.hstack((X_train, X_control))
    else:
        n_nuissance = 0

    # Standardize data
    X_train, X_test = scale(X_train, X_test)
    y_train, y_test = scale(y_train, y_test)

    # Find alpha
    alpha = inner_ridge(X_train, y_train)
    print(f'alpha = {alpha}')

    # Fit the regression
    model, betas = outer_ridge(X_train, y_train, alpha)

    # Prediction
    if inds is None:
        if predict_by_feature:
            y_pred, betas = predict(X_test, y_train, model.coef_, n_nuissance, by_feature=True)
        else:
            y_pred, betas = predict(X_test, y_train, model.coef_, n_nuissance)
    else:
        y_pred, betas = predict_multi_feature(X_test, y_train, model.coef_, inds)
    return y_pred, y_test, model.coef_
