from itertools import product
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def compute_distance(arr):
    a = arr[::2]
    b = arr[1::2]
    if len(a) > len(b):
        a = a[1:]
    return euclidean(a, b)


def feature2color(key=None):
    d = dict()
    d['indoor'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['spatial expanse'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
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


process = 'BehavioralSummary'
data_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw'
output_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
Path(f'{output_dir}/{process}').mkdir(parents=True, exist_ok=True)
Path(f'{figure_dir}/{process}').mkdir(parents=True, exist_ok=True)
features = ['spatial expanse', 'object', 'agent distance', 'facingness', 'joint action', 'communication', 'valence', 'arousal']
feature_map = {'expanse': 'spatial expanse', 'joint': 'joint action',
               'distance': 'agent distance', 'communicating': 'communication'}

data = pd.read_csv(f'{data_dir}/annotations/individual_subject_ratings.csv')
txt_output = open(f'{output_dir}/{process}/summary.txt', 'w')
txt_output.write(f'n subjects: {data.subjectID.max()} \n ')

average = data.drop(columns=['subjectID']).groupby(['video_name', 'question_name']).mean().reset_index(
    drop=False).rename(columns={'likert_response': 'average'})
standard_deviation = data.drop(columns=['subjectID']).groupby(['video_name', 'question_name']).std().reset_index(
    drop=False).rename(columns={'likert_response': 'standard_deviation'})
n = data.drop(columns=['subjectID']).groupby(['video_name', 'question_name']).count().reset_index(drop=False).rename(
    columns={'likert_response': 'n_responses'})
summary = average.merge(standard_deviation).merge(n)

txt_output.write(f'most responses: {summary.n_responses.max()} \n ')
txt_output.write(f'least responses: {summary.n_responses.min()} \n ')
txt_output.write(f'average response: {summary.n_responses.mean():.2f} \n ')
txt_output.write(f'std of responses: {summary.n_responses.std():.2f} \n ')

summary["rho"] = 0
summary = summary.set_index(['video_name', 'question_name'])

n_shuffles = 100
videos = list(data.video_name.unique())
questions = list(data.question_name.unique())
for v, q in tqdm(product(videos, questions)):
    responses = data.loc[(data.video_name == v) & (data.question_name == q), 'likert_response'].to_numpy()
    distance = []
    for i in range(n_shuffles):
        np.random.shuffle(responses)
        distance.append(compute_distance(responses))
    summary.loc[(v, q), "distance"] = np.array(distance).sum() / n_shuffles

summary.to_csv(f'{output_dir}/{process}/behavioral_summary.csv')
summary = summary.reset_index(drop=False)
summary = summary.replace(feature_map)
summary['feature'] = pd.Categorical(summary.question_name, ordered=True, categories=features)

font = 6
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context='paper', style='white', rc=custom_params)
_, ax = plt.subplots(figsize=(3, 3))

sns.barplot(x='feature', y='distance', data=summary, ax=ax, color='gray')
for bar, feature in zip(ax.patches, features):
    color = feature2color(feature)
    bar.set_color(color)

ax.set_xlabel('')
ax.set_ylabel(f'Euclidean distance', fontsize=font+2)
ax.set_xticklabels(features,
                   fontsize=font,
                   rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{figure_dir}/{process}/reliability.pdf')
txt_output.close()
