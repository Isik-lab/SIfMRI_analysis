import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt


def feature2color(key=None):
    d = dict()
    d['indoor'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['expanse'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['object'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['agent distance'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['facingness'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['joint action'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['communication'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['valence'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    d['arousal'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    if key is not None:
        return d[key]
    else:
        return d


# low_com = 'flickr-8-7-2-6-6-0-0-8-3387266008_18.mp4'
# high_com = 'flickr-7-7-0-5-0-7-0-9-4177050709_33.mp4'

low_com = 'yt-KZqqB7yoVYw_11.mp4'
high_com = 'flickr-4-9-9-9-8-3-1-6-25549998316_29.mp4'

data_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis'
process = 'ExampleRatings'
figure_dir = f'{data_dir}/reports/figures/{process}'
Path(figure_dir).mkdir(exist_ok=True, parents=True)

df = pd.read_csv(f'{data_dir}/data/raw/annotations/annotations.csv')
features = ['indoor', 'expanse', 'object', 'agent distance',
            'facingness', 'joint action', 'communication', 'valence', 'arousal']
df.drop(columns=['cooperation', 'dominance', 'intimacy'], inplace=True)
df.rename(columns={'transitivity': 'object'}, inplace=True)

for i, feature in enumerate(features):
    df.rename(columns={feature: f'Rating{i}'}, inplace=True)

df = pd.wide_to_long(df, stubnames='Rating', i='video_name', j='feature').reset_index(drop=False)
for i, feature in enumerate(features):
    df.feature.replace({i: feature}, inplace=True)
print(df.feature.unique())

for vid in [low_com, high_com]:
    cur_df = df.loc[df.video_name == vid]
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='paper', style='white', rc=custom_params)
    _, ax = plt.subplots(1, figsize=(1.75, 1.5))

    sns.barplot(x='feature', y='Rating', data=cur_df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylim([0, 1])

    # Change the ytick font size
    label_format = '{:.1f}'
    plt.locator_params(axis='y', nbins=3)
    y_ticklocs = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
    ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=6)
    ax.set_ylabel('Rating', fontsize=6)

    ax.set_xticklabels(features, fontsize=6)

    # Change the xaxis font size and colors
    ax.set_xticklabels(features,
                       rotation=45, ha='right')
    # for ticklabel, pointer in zip(features, ax.get_xticklabels()):
    #     color = feature2color(ticklabel)
    #     # color[-1] = 1.
    #     pointer.set_color(color)
    #     pointer.set_weight('bold')

    # Manipulate the color and add error bars
    for bar, feature in zip(ax.patches, features):
        color = feature2color(feature)
        bar.set_color(color)
    ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{vid.replace('mp4', 'svg')}")
