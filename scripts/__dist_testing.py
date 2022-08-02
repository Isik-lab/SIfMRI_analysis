import numpy as np
import matplotlib.pyplot as plt


def corr(x, y):
 x_m = x - x.mean()
 y_m = y - y.mean()

 numer = np.sum(x_m*y_m)
 denom = np.sum(x_m*x_m) * np.sum(y_m * y_m)
 return numer / np.sqrt(denom)

def perm(a, b, n_perm=10000):
 r_null = np.zeros(n_perm)
 for i in range(n_perm):
  inds = np.random.default_rng(i).permutation(a.size)
  r_null[i] = corr(a[inds], b)
 return r_null

for i, std in enumerate(np.linspace(1e-4, 1e3, num=5)):
 print(f'std = {std}')
 r = 0.
 while r <= 0:
   y = np.random.normal(loc=1, scale=std, size=50)
   y_hat = np.random.normal(loc=1, scale=std, size=50)
   r = corr(y, y_hat)
 r_null = perm(y, y_hat)

 _, ax = plt.subplots()
 ax.hist(r_null)
 ys = np.arange(0, 5000)
 ax.plot(np.ones_like(ys)*r, ys, 'r--')
 plt.savefig(f'nulls/{i}.png')

 print(f'mean = {r_null.mean()}, std = {r_null.std()}')
 print(f'r = {r:.2f}')
 print()