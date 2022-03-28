import numpy as np
from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
import itertools
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf_roi
import nibabel as nib
import seaborn as sns


def custom_palette(rgb=True):
    colors = dict()
    if rgb:
        colors['mustard'] = tuple(np.array([245., 221., 64.]) / 256)
        colors['reddish'] = tuple(np.array([218., 83., 91.]) / 256)
        colors['purple'] = tuple(np.array([133., 88., 244.]) / 256)
        colors['cyan'] = tuple(np.array([115., 210., 223.]) / 256)
        colors['blue'] = tuple(np.array([105., 150., 237.]) / 256)
    else:
        colors['mustard'] = '#F5DD40'
        colors['reddish'] = '#DA535B'
        colors['purple'] = '#8558F4'
        colors['cyan'] = '#73D2DF'
        colors['blue'] = '#6796ED'
    return colors


def feature_categories():
    d = dict()
    d['indoor'] = 'scene'
    d['expanse'] = 'scene'
    d['transitivity'] = 'object'
    d['agent distance'] = 'social primitive'
    d['facingness'] = 'social primitive'
    d['joint action'] = 'social'
    d['communication'] = 'social'
    d['cooperation'] = 'social'
    d['dominance'] = 'social'
    d['intimacy'] = 'social'
    d['valence'] = 'social'
    d['arousal'] = 'social'
    return d


def feature_colors():
    d = dict()
    d['indoor'] = 'mustard'
    d['expanse'] = 'mustard'
    d['transitivity'] = 'reddish'
    d['agent distance'] = 'purple'
    d['facingness'] = 'purple'
    d['joint action'] = 'cyan'
    d['communication'] = 'cyan'
    d['cooperation'] = 'cyan'
    d['dominance'] = 'cyan'
    d['intimacy'] = 'cyan'
    d['valence'] = 'cyan'
    d['arousal'] = 'cyan'
    return d


def custom_seaborn_cmap():
    colors = custom_palette(rgb=False)
    colors = list(colors.values())
    palette = sns.color_palette(colors, as_cmap=True)
    return palette


def custom_nilearn_cmap():
    palette = custom_palette()
    colors = feature_colors()
    cmap = sns.color_palette('Paired', len(colors), as_cmap=True)
    out_colors = []
    for i, feature in enumerate(colors.keys()):
        color = colors[feature]
        rgb = palette[color]
        out_colors.append(rgb)
        cmap._lut[i] = list(rgb) + [1.]
    cmap.colors = tuple(out_colors)

    cmap._lut[cmap._i_over] = [0., 0., 0., 0.]
    cmap._lut[cmap._i_under] = [0., 0., 0., 0.]
    cmap._lut[cmap._i_bad] = [0., 0., 0., 0.]
    return cmap


def mkNifti(arr, mask, im, nii=True):
    out_im = np.zeros(mask.size, dtype=arr.dtype)
    inds = np.where(mask)[0]
    out_im[inds] = arr
    if nii:
        out_im = out_im.reshape(im.shape)
        out_im = nib.Nifti1Image(out_im, affine=im.affine)
    return out_im


def get_vmax(texture):
    array = np.hstack((texture['left'], texture['right']))
    i = np.where(~np.isclose(array, 0))
    return array[i].mean() + (3 * array[i].std())


def _colorbar_from_array(array, threshold, cmap, vmax=None):
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
    if threshold is None:
        threshold = 0.

    if vmax is None:
        vmax = np.nanmax(array)
    norm = Normalize(vmin=threshold, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # set colors to grey for absolute values < threshold
    istart = int(threshold)
    istop = int(norm(threshold, clip=True) * (cmap.N - 1))
    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.)
    our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                 cmaplist, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=our_cmap,
                               norm=plt.Normalize(vmin=threshold, vmax=vmax))
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
                      alpha=1,
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
        sm = _colorbar_from_array(array, threshold, get_cmap(cmap), vmax)

        cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    if title is not None:
        fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)


def plot_ROI_results(df, out_name, variable, noise_ceiling=None):
    features = df.Features.unique()
    n_features = len(features)

    # Set up figures
    sns.set(style='whitegrid', context='talk', rc={'figure.figsize': (6, 5)})
    fig, ax = plt.subplots()
    sns.barplot(x='Features', y=variable,
                data=df, ax=ax,
                hue='Feature category',
                palette=custom_seaborn_cmap(),
                dodge=False, ci=None)

    # Plot noise ceiling
    if noise_ceiling is not None:
        x = np.linspace(-0.5, n_features-0.5, num=3)
        y1 = np.ones_like(x) * noise_ceiling
        ax.plot(x, y1, color='gray', alpha=0.5, linewidth=3)

    for ifeature, feature in enumerate(features):
        x = ifeature
        sig = df.loc[df.Features == feature, 'group sig'].reset_index(drop=True)[0]
        p = df.loc[df.Features == feature, 'group_pcorrected'].reset_index(drop=True)[0]
        if sig:
            if p > 0.01:
                text = '*'
            elif p < 0.01 and p > 0.001:
                text = '**'
            elif p < 0.001:
                text = '***'
            ax.annotate(text, (x, 0.5), fontsize=20,
                        weight='bold', ha='center', color='gray')

        y1 = df.loc[df.Features == feature, 'low sem'].mean()
        y2 = df.loc[df.Features == feature, 'high sem'].mean()
        plt.plot([x, x], [y1, y2], 'black', linewidth=2)

    # #Aesthetics
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ax.set_xlabel('')
    ax.set_ylabel('Prediction accuracy ($\it{r}$)')
    ax.set_ylim([-0.1, 0.78])
    sns.despine(left=True)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()
