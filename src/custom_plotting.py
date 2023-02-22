import nilearn.image
import numpy as np
from matplotlib import gridspec, ticker
import matplotlib as mpl
from matplotlib.colors import Normalize, LinearSegmentedColormap
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting, surface
from nilearn.plotting import plot_surf_roi, plot_surf_contours, plot_surf_stat_map
import nibabel as nib
import seaborn as sns


def custom_palette(rgb=True):
    colors = dict()
    if rgb:
        colors['white'] = tuple(np.array([256., 256., 256.]) / 256)
        colors['gray'] = tuple(np.array([225., 225., 225.]) / 256)
        colors['black'] = tuple(np.array([50., 50., 50.]) / 256)
        colors['bluegray'] = tuple(np.array([149., 167., 228.]) / 256)
        colors['mustard'] = tuple(np.array([245., 221., 64.]) / 256)
        colors['purple'] = tuple(np.array([133., 88., 244.]) / 256)
        colors['cyan'] = tuple(np.array([115., 210., 223.]) / 256)
        colors['reddish'] = tuple(np.array([218., 83., 91.]) / 256)
    else:
        colors['gray'] = '#808080'
        colors['bluegray'] = '#95A7E4'
        colors['mustard'] = '#F5DD40'
        colors['purple'] = '#8558F4'
        colors['cyan'] = '#73D2DF'
        colors['reddish'] = '#DA535B'
    return colors


def feature_categories():
    d = dict()
    d['AlexNet-conv2'] = 'low-level model'
    d['motion energy'] = 'low-level model'
    d['indoor'] = 'scene & object'
    d['expanse'] = 'scene & object'
    d['object'] = 'scene & object'
    d['agent distance'] = 'social primitive'
    d['facingness'] = 'social primitive'
    d['joint action'] = 'social interaction'
    d['communication'] = 'social interaction'
    d['valence'] = 'affective'
    d['arousal'] = 'affective'
    return d

def category_colors():
    d = dict()
    d['filler'] = 'white'
    d['AlexNet-conv2'] = 'black'
    d['motion energy'] = 'bluegray'
    d['scene & object'] = 'mustard'
    d['social primitive'] = 'purple'
    d['social interaction'] = 'cyan'
    d['affective'] = 'reddish'
    return d

def custom_nilearn_cmap():
    palette = custom_palette()
    colors = category_colors()
    cmaplist = [palette[colors[category]] + (1.,) for category in colors.keys()]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, len(colors))

    # # cmap = sns.color_palette('Paired', len(colors), as_cmap=True)
    # out_colors = []
    # for i, category in enumerate(colors.keys()):
    #     color = colors[category]
    #     rgb = palette[color]
    #     out_colors.append(rgb)
    #     cmap._lut[i] = list(rgb) + [1.]
    #     cmap._lut[i+1] = list(rgb) + [1.]
    # cmap.colors = tuple(out_colors)
    #
    # cmap._lut[cmap._i_over] = [0., 0., 0., 0.]
    # cmap._lut[cmap._i_under] = [0., 0., 0., 0.]
    # cmap._lut[cmap._i_bad] = [0., 0., 0., 0.]
    return cmap


def custom_pca_cmap(n):
    # initialize a cmap object
    cmap = sns.color_palette('Paired', n)
    cmap = ListedColormap(cmap)

    palette = sns.color_palette("cubehelix", n)

    out_colors = []
    for i, rgb in enumerate(palette):
        if i == n:
            break
        out_colors.append(rgb)
    cmap.colors = tuple(out_colors)
    return cmap


def custom_preference_cmap():
    palette = [[0.95703125, 0.86328125, 0.25],  # mustard
               [0.57421875, 0.51796875, 0.15],  # mustard
               [0.8515625, 0.32421875, 0.35546875],  # red
               [0.51953125, 0.34375, 0.953125],  # purple
               [0.31171875, 0.20625, 0.571875],  # purple
               [0.44921875, 0.8203125, 0.87109375],  # cyan
               [0.35938, 0.65625, 0.69688],
               [0.2875, 0.525, 0.5575],
               [0.23, 0.42, 0.446],
               [0.184, 0.336, 0.3568],
               [0.1472, 0.2688, 0.28544],
               [0.11776, 0.21504, 0.22835]]

    # initialize a discrete cmap object
    cmap = sns.color_palette('Paired', len(palette))
    cmap = ListedColormap(cmap)
    cmap.colors = tuple(palette)
    return cmap


def mkNifti(arr, mask, im, nii=True, fill=0):
    def fill_arr(dims, dtype, fill):
        if fill == 0:
            a = np.zeros(dims, dtype=dtype)
        else:
            a = np.ones(dims, dtype=dtype)*fill
        return a

    mask = mask.astype('bool')
    if arr.ndim == 1:
        out_im = fill_arr(mask.size, arr.dtype, fill)
        out_im[mask] = arr
    else:
        out_im = fill_arr((mask.size,) + arr.shape[1:], arr.dtype, fill)
        out_im[mask, ...] = arr

    if nii:
        out_im = out_im.reshape(im.shape)
        out_im = nib.Nifti1Image(out_im, affine=im.affine)
    return out_im


def get_vmax(texture):
    array = np.hstack((texture['left'], texture['right']))
    i = np.where(~np.isclose(array, 0))
    return array[i].mean() + (3 * array[i].std())


def _colorbar_from_array(cmap, vmax):
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
    cmap = plt.get_cmap(cmap, vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                 cmaplist, cmap.N)

    vmax = np.round(vmax, decimals=1)
    norm = plt.Normalize(vmin=0., vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=our_cmap,
                               norm=norm)
    # # fake up the array of the scalar mappable.
    sm._A = []
    return sm


def roi_paths(hemi):
    if hemi == 'left':
        h = 'l'
    else:
        h = 'r'
    d = dict()
    topdir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw/group_parcels/'
    d['STS'] = f'{topdir}/BenDeen_fROIs/{h}STS.nii.gz'
    d['STS'] = f'{topdir}/BenDeen_fROIs/{h}STS.nii.gz'
    d['SIpSTS'] = f'{topdir}/rPSTS_SI_parcel/SocialInteract_rpSTS_vol.nii.gz'
    d['LOC'] = f'{topdir}/kanwisher_gss/object_parcels/{h}LOC.img'
    d['PPA'] = f'{topdir}/kanwisher_gss/scene_parcels/{h}PPA.img'
    d['OPA'] = f'{topdir}/kanwisher_gss/scene_parcels/{h}TOS.img'
    d['EBA'] = f'{topdir}/kanwisher_gss/body_parcels/{h}EBA.img'
    d['FFA'] = f'{topdir}/kanwisher_gss/face_parcels/{h}FFA.img'
    d['TPJ'] = f'{topdir}/fROItMaps/{h.capitalize()}TPJ_wxyz.img'
    d['MT'] = f'{topdir}/SabineKastner/subj_vol_all/perc_VTPM_vol_roi13_{h}h.nii.gz'
    d['V1v'] = f'{topdir}/SabineKastner/subj_vol_all/perc_VTPM_vol_roi1_{h}h.nii.gz'
    d['V1d'] = f'{topdir}/SabineKastner/subj_vol_all/perc_VTPM_vol_roi2_{h}h.nii.gz'
    d['V2v'] = f'{topdir}/SabineKastner/subj_vol_all/perc_VTPM_vol_roi3_{h}h.nii.gz'
    d['V2d'] = f'{topdir}/SabineKastner/subj_vol_all/perc_VTPM_vol_roi4_{h}h.nii.gz'
    return d


def load_parcellation(fsaverage, roi, hemi):
    paths = roi_paths(hemi)
    if roi == 'pSTS' and hemi == 'left':
        parcellation = None
    else:
        vol = nib.load(paths[roi])
        parcellation = surface.vol_to_surf(vol, fsaverage[f'pial_{hemi}'], interpolation='nearest')
        parcellation[parcellation != 0] = 1
        parcellation.astype('int')
    return parcellation


def _colorbar_betas(cmap, vmax):
    cmap = plt.get_cmap(cmap)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                 cmaplist, cmap.N)

    vmax = np.round(vmax, decimals=1)
    norm = plt.Normalize(vmin=-1*vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=our_cmap,
                               norm=norm)

    # # fake up the array of the scalar mappable.
    sm._A = []
    return sm


def plot_betas(fsaverage, texture,
               roi=None,
               title=None,
               vmax=None,
               modes=['lateral', 'medial', 'ventral'],
               hemis=['left', 'right'],
               cmap=None, threshold=0.01,
               output_file=None, colorbar=True,
               cbar_title=r"$\beta$ weights",
               kwargs={}):
    if vmax is None:
        array = np.hstack((texture['left'], texture['right']))
        vmax = array.max().round(decimals=2)
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
        plot_surf_stat_map(surf_mesh=fsaverage[f'infl_{hemi}'],
                           stat_map=texture[hemi],
                           view=mode, hemi=hemi,
                           bg_map=bg_map,
                           alpha=1.,
                           axes=ax,
                           colorbar=False,  # Colorbar created externally.
                           vmax=vmax,
                           threshold=threshold,
                           cmap=cmap,
                           symmetric_cbar=True,
                           **kwargs)
        rect = ax.patch
        rect.set_facecolor('white')

        if roi:
            colors = [['white'], ['black'], ['gray']]
            for ir, r in enumerate(roi):
                parcellation = load_parcellation(fsaverage, r, hemi)
                if parcellation is not None:
                    plot_surf_contours(fsaverage[f'infl_{hemi}'], parcellation, labels=[r],
                                       levels=[1], axes=ax, legend=False,
                                       colors=colors[ir])
        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        ax.dist = 7

    if colorbar:
        sm = _colorbar_betas(cmap, vmax)

        cbar_grid = gridspec.GridSpecFromSubplotSpec(2, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        bar_max = vmax-.01
        ticks = np.linspace((-1*bar_max), bar_max, num=3).round(decimals=2)
        cbar = fig.colorbar(sm, cax=cbar_ax,
                            orientation='horizontal', ticks=ticks)
                            # label=cbar_title)
        cbar.set_label(label=cbar_title, size=34)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(30)

    if title is not None:
        fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)


def plot_surface_stats(fsaverage, texture,
                       roi=None,
                       title=None,
                       modes=['lateral', 'medial', 'ventral'],
                       hemis=['left', 'right'],
                       cmap=None, threshold=0.01,
                       output_file=None, colorbar=True,
                       vmax=None,
                       cbar_title=r"Correlation ($r$)",
                       kwargs={}):
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
                      alpha=1.,
                      axes=ax,
                      colorbar=False,  # Colorbar created externally.
                      vmax=vmax,
                      threshold=threshold,
                      cmap=cmap,
                      **kwargs)

        rect = ax.patch
        rect.set_facecolor('white')

        if roi:
            for r in roi:
                parcellation = load_parcellation(fsaverage, r, hemi)
                if parcellation is not None:
                    plot_surf_contours(fsaverage[f'infl_{hemi}'], parcellation, labels=[r],
                                       levels=[1], axes=ax, legend=False,
                                       colors=['white'])
        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        ax.dist = 7

    if colorbar:
        sm = _colorbar_from_array(cmap, vmax)

        cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        ticks = np.linspace(0, vmax, num=3).round(decimals=2)
        cbar = fig.colorbar(sm, cax=cbar_ax,
                            orientation='horizontal', ticks=ticks)
        cbar.set_label(label=cbar_title, size=26)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(21)

    if title is not None:
        fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

    if output_file is not None:
        fig.savefig(output_file, facecolor='white')
        plt.close(fig)


def plot_ROI_results(df, out_name, variable, noise_ceiling=None,
                     ylabel=None):
    if ylabel is None:
        ylabel = variable
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
        x = np.linspace(-0.5, n_features - 0.5, num=3)
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
            ax.annotate(text, (x, 0.45), fontsize=20,
                        weight='bold', ha='center', color='gray')

        y1 = df.loc[df.Features == feature, 'low sem'].mean()
        y2 = df.loc[df.Features == feature, 'high sem'].mean()
        plt.plot([x, x], [y1, y2], 'black', linewidth=2)

    # #Aesthetics
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_ylim([-0.2, 0.78])
    sns.despine(left=True)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()
