import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

n = 3
data_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis'
process = 'ExampleRatings'
figure_dir = f'{data_dir}/reports/figures/{process}'
raw_dir = f'{data_dir}/data/raw'
Path(figure_dir).mkdir(exist_ok=True, parents=True)

df = pd.read_csv(f'{raw_dir}/annotations/annotations.csv')
features = ['expanse', 'object', 'agent distance',
            'facingness', 'joint action', 'communication', 'valence', 'arousal']
df.drop(columns=['cooperation', 'dominance', 'intimacy'], inplace=True)
df.rename(columns={'transitivity': 'object'}, inplace=True)

custom_params = {"axes.spines.right": False, "axes.spines.top": False,
                 "axes.spines.left": False, "axes.spines.bottom": False}
sns.set_theme(context='poster', style='white', rc=custom_params)

_, ax = plt.subplots(len(features), n, figsize=(4.5, 11))
for ifeature, feature in enumerate(features):
    vids = df.sort_values(by=feature, ascending=True).reset_index(drop=True)
    for i in range(0, n):
        im = Image.open(f"{raw_dir}/images/{vids.iloc[i].video_name.replace('mp4', 'jpg')}")
        ax[ifeature, i].imshow(im)
        ax[ifeature, i].set_axis_off()
plt.savefig(f'{figure_dir}/low_ratings.pdf', bbox_inches='tight')

_, ax = plt.subplots(len(features), n, figsize=(4.5, 11))
for ifeature, feature in enumerate(features):
    vids = df.sort_values(by=feature, ascending=False).reset_index(drop=True)
    for i in range(0, n):
        im = Image.open(f"{raw_dir}/images/{vids.iloc[i].video_name.replace('mp4', 'jpg')}")
        ax[ifeature, i].imshow(im)
        ax[ifeature, i].set_axis_off()
plt.savefig(f'{figure_dir}/high_ratings.pdf', bbox_inches='tight')


