import numpy as np
        
def corr2d(x, y):
    x_m = x - x.mean(axis=0)
    y_m = y - y.mean(axis=0)

    r = np.zeros(y_m.shape[0])
    for i in range(y_m.shape[0]): 
        r[i] = (x_m[i, :] @ y_m[i, :]) / (np.sqrt((x_m[i, :] @ x_m[i, :]) * (y_m[i, :] @ y_m[i, :]))) 
    return r

def corr1d(x, y):
    """
        input:
            x:
            y:
        output:
    """
    x_m = x - x.mean()
    y_m = y - y.mean()

    return (x_m @ y_m) / (np.sqrt((x_m @ x_m) * (y_m @ y_m))) 

def permutation_test(self, a, b, test_inds=None,
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

def bootstrap(a, b, test_inds, n_samples=int(5e3)):
    r_var = np.zeros(n_samples)
    for i in range(n_samples):
        inds = np.random.default_rng(i).choice(np.arange(test_inds.shape[0]), size=test_inds.shape[0])
        inds = test_inds[inds, :].flatten()
        r_var[i] = corr1d(a[inds], b)
    return r_var
            
def mask(path, stat_map):
    from nilearn import image
    
    #activity in ROI
    mask = image.load_img(path)
    mask = np.array(mask.dataobj, dtype='bool').flatten()
    roi_activation = stat_map[mask, :]

    #Remove nan values (these are voxels that do not vary across the different videos)
    inds = ~np.any(np.isnan(roi_activation), axis=1)
    return roi_activation[inds, ...]

def regression(X_train_, y_train_, X_test_, features): 
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, RidgeCV
    
    #Standardize X
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)

    #Ridge CV to get the alpha parameter
    clf = RidgeCV(cv=4).fit(X_train_, y_train_)

    #Fit the Ridge model with the found best alpha
    lr = Ridge(fit_intercept=False, alpha=clf.alpha_).fit(X_train_, y_train_)

    #Scale testing 
    mean = scaler.mean_[:X_test_.shape[1]]
    var = scaler.var_[:X_test_.shape[1]]
    X_test_ = (X_test_ - mean)/var
    
    y_pred = np.zeros((len(features), X_test.shape[0]))
    for ifeature, feature in enumerate(features):
        y_pred[ifeature, :] = X_test_[:, ifeature] * lr.coef_[ifeature]
    return y_pred
    