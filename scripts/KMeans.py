import os
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib

from src.regress import ridge
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist


def mkNifti(arr, mask, im, nii=True):
    out_im = np.zeros(mask.size, dtype=arr.dtype)
    inds = np.where(mask)[0]
    out_im[inds] = arr
    if nii:
        out_im = out_im.reshape(im.shape)
        out_im = nib.Nifti1Image(out_im, affine=im.affine)
    return out_im


def silhouette_plot(n_clusters, cluster_labels,
                    silhouette_avg, sample_silhouette_values,
                    centers):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f'{figure_dir}/silhoutte_nclusters-{n_clusters}.pdf')


process = 'KMeans'
top_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis'
data_dir = f'{top_dir}/data/raw'
out_dir = f'{top_dir}/data/interim'
figure_dir = f'{top_dir}/reports/figures/{process}'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

# Load mask
reliability = np.load(f'{out_dir}/Reliability/sub-all_reliability-mask.npy').astype('bool')
im = nib.load(f'{out_dir}/Reliability/sub-all_stat-rho_statmap.nii.gz')

rs = np.load(f'{out_dir}/VoxelPermutation/sub-all/sub-all_feature-all_rs-filtered.npy').astype('bool')
rs = mkNifti(rs, reliability, im, nii=False)
mask = np.logical_and(reliability, rs)

X = []
n_voxels = sum(mask)
for sid_ in range(4):
    sid = str(sid_ + 1).zfill(2)
    preproc_betas = np.load(f'{out_dir}/GroupRuns/sub-{sid}/sub-{sid}_train-data.npy')

    # Filter the beta values to the reliable voxels
    preproc_betas = preproc_betas[mask, :]

    # Mean center the activation within subject
    # offset_subject = betas.mean()
    # betas -= offset_subject

    if type(X) is list:
        X = preproc_betas.T
    else:
        X += preproc_betas.T
X /= 4

df = pd.read_csv(f'{data_dir}/annotations/annotations.csv')
train = pd.read_csv(f'{data_dir}/annotations/train.csv')
df = df.merge(train)
df = df.sort_values(by=['video_name']).drop(columns=['video_name'])
y = df.to_numpy()

coef, _ = ridge(X, y)
X = coef.T

inertias = []
distortions = []
stop = 12
n = stop - 1
for i_cluster, n_clusters in tqdm(enumerate(range(2, stop)), total=n):
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(X)
    # These lines are slow
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    silhouette_plot(n_clusters, cluster_labels,
                    silhouette_avg, sample_silhouette_values,
                    kmeans.cluster_centers_)

    distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeans.inertia_)

    fig, (ax1, ax2,) = plt.subplots(2, 1, sharex=True)
    ax1.plot(np.arange(2, i_cluster + 3), distortions, '.-')
    ax2.set_ylabel('Distortions')
    ax2.plot(np.arange(2, i_cluster + 3), inertias, '.-')
    ax2.set_ylabel('Inertia')
    ax2.set_xlabel('Number of clusters')
    plt.tight_layout()
    plt.savefig(f'{figure_dir}/elbow_plot.pdf')
