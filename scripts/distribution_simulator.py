import numpy as np
import matplotlib.pyplot as plt
from src.tools import calculate_p


def corr(x, y):
    x_m = x - x.mean()
    y_m = y - y.mean()

    numer = np.sum((x_m * y_m))
    denom = np.sqrt(np.sum(x_m * x_m) * np.sum(y_m * y_m))
    return numer / denom


n_perm = 10000
stds = [1, 10, 100]
colors = ['r', 'g', 'b']
n_sample = 200
r = 0.12
for std, color in zip(stds, colors):
    arr1 = np.random.randint(low=-1*std, high=std, size=n_sample)
    arr2 = np.random.randint(low=-1*std, high=std, size=n_sample)
    r_null = np.zeros(n_perm)
    for i in range(n_perm):
        inds = np.random.default_rng(i).permutation(n_sample)
        r_null[i] = corr(arr1[inds], arr2)
    plt.hist(r_null, color=color, label=std)
    r_crit = np.percentile(r_null, 95)
    print(f'color = {color}, critical r = {r_crit:.4f}')
    print(f'r = {r}, r_crit < r = {r_crit < r}')
    print(f'p = {calculate_p(r_null, r, n_perm, "greater"):.4f}')
plt.title(f'n_sample = {n_sample}, std = {r_null.std():.2f}')
plt.legend()
plt.show()