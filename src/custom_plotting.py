import numpy as np
from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
import itertools
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf_roi
import nibabel as nib

def mkNifti(arr, mask, im, nii=True):
    out_im = np.zeros(mask.size, dtype=arr.dtype)
    inds = np.where(mask)[0]
    out_im[inds] = arr
    if nii:
        out_im = out_im.reshape(im.shape)
        out_im = nib.Nifti1Image(out_im, affine=im.affine)
    return out_im

def _colorbar_from_array(array, threshold, cmap):
    """Generate a custom colorbar for an array.
    Internal function used by plot_img_on_surf
    array : np.ndarray
        Any 3D array.
    vmax : float
        upper bound for plotting of stat_map values.
    threshold : float
        If None is given, the colorbar is not thresholded.
        If a number is given, it is used to threshold the colorbar.
        Absolute values lower than threshold are shown in gray.
    kwargs : dict
        Extra arguments passed to _get_colorbar_and_data_ranges.
    cmap : str, optional
        The name of a matplotlib or nilearn colormap.
        Default='cold_hot'.
    """
    vmin = array.min()
    vmax = array.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if threshold is None:
        threshold = 0.

    # set colors to grey for absolute values < threshold
    istart = int(vmin)
    istop = int(norm(threshold, clip=True) * (cmap.N - 1))
    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.)
    our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                 cmaplist, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=our_cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable.
    sm._A = []

    return sm

def plot_surface_stats(fsaverage, texture,
                       title=None,
                       modes=['lateral', 'medial', 'ventral'],
                       hemis=['left', 'right'],
                       cmap=None, threshold=0.01,
                       output_file=None, colorbar=True,
                       vmax=None, kwargs={}):
    cbar_h = .25
    title_h = .25 * (title is not None)
    # Set the aspect ratio, but then make the figure twice as big to increase resolution
    w, h = plt.figaspect((len(modes) + cbar_h + title_h) / len(hemis)) * 2
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    height_ratios = [title_h] + [1.] * len(modes) + [cbar_h]
    grid = gridspec.GridSpec(
        len(modes) + 2, len(hemis),
        left=0., right=1., bottom=0., top=1.,
        height_ratios=height_ratios, hspace=0.0, wspace=0.0)
    axes = []
    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):
        bg_map = fsaverage['sulc_%s' % hemi]
        ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
        axes.append(ax)
        plot_surf_roi(surf_mesh=fsaverage[f'infl_{hemi}'],
                      roi_map=texture[hemi],
                      view=mode, hemi=hemi,
                      bg_map=bg_map,
                      alpha=0.5,
                      axes=ax,
                      colorbar=False,  # Colorbar created externally.
                      vmax=vmax,
                      threshold=threshold,
                      cmap=cmap,
                      **kwargs)
        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        ax.dist = 7

    if colorbar:
        array = np.hstack((texture['left'], texture['right']))
        sm = _colorbar_from_array(array, threshold, get_cmap(cmap))

        cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    if title is not None:
        fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
