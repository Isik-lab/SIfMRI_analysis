#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
from pathlib import Path
import tqdm
import os

import pandas as pd
import numpy as np

import imageio
from PIL import Image

import torchvision.models as models
from matplotlib import pyplot as plt
from torchvision import transforms as trn
from torch.autograd import Variable as V
import torch.nn as nn
from sklearn.decomposition import PCA


def combine(X, combination=None):
    X = X.data.numpy()

    # Take the mean of the spatial axes
    if X.ndim > 2 and combination is not None:
        if combination == 'mean':
            X = X.mean(axis=(X.ndim-2, X.ndim-1))
        elif combination == 'max':
            X = X.max(axis=(X.ndim-2, X.ndim-1))
    return X.flatten()


def preprocess(image_fname, resize=256):
    center_crop = trn.Compose([
        trn.Resize((resize, resize)),
        #trn.CenterCrop(crop),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if isinstance(image_fname, str):
        img_input = Image.open(image_fname)
    else:
        img_input = image_fname
    return V(center_crop(img_input).unsqueeze(0))


class AlexNet_Extractor(nn.Module):
    def __init__(self, net):
        super(AlexNet_Extractor, self).__init__()
        self.net = net

    def forward(self, img, layer, combination=None):
        if layer < 6:
            #conv layers activation
            model = self._get_features(layer)
            model.eval()
            X = model(img)
        elif layer >= 6 and layer < 8:
            #fc activations
            model = copy.deepcopy(self.net)
            model.classifier = self._get_classifier(layer)
            model.eval()
            X = model(img)
        elif layer == 8:
            #class activations layer
            self.net.eval()
            X = self.net(img)
        return combine(X, combination)

    def _get_features(self, layer):
        switcher = {
            1: 3,   # from features
            2: 6,
            3: 8,
            4: 10,
            5: 13}
        index = switcher.get(layer)
        features = nn.Sequential(
            # stop at the layer
            *list(self.net.features.children())[:index]
        )
        return features

    def _get_classifier(self, layer):
        switcher = {6: 3,   # from classifier
                    7: 6}
        index = switcher.get(layer)
        classifier = nn.Sequential(
            # stop at the layer
            *list(self.net.classifier.children())[:index]
        )
        return classifier


class AlexNetActivations():
    def __init__(self, args):
        self.process = 'AlexNetActivations'
        self.set = args.set
        self.layer = args.layer
        self.overwrite = args.overwrite
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

    def get_PCs(self, X):
        pca = PCA(whiten=False, n_components=0.8)
        pca.fit(X)
        _, axes = plt.subplots(2)
        axes[0].plot(pca.explained_variance_ratio_, '-o')
        axes[0].set_title('Explained variance')
        axes[1].plot(pca.explained_variance_ratio_.cumsum(), '-o')
        axes[1].set_title('Cumulative explained variance')
        plt.savefig(f'{self.figure_dir}/pca_visualization_set-{self.set}.pdf')

    def run(self):
        out_file = f'{self.out_dir}/alexnet_conv{self.layer}_set-{self.set}_avgframe.npy'
        if not os.path.exists(out_file) or self.overwrite:
            df = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
            df.sort_values(by=['video_name'], inplace=True)
            vid_dir = f'{self.data_dir}/videos'

            model = models.alexnet(weights='IMAGENET1K_V1')
            model.eval()
            feature_extractor = AlexNet_Extractor(model)

            activation = []
            for vid in tqdm.tqdm(df.video_name, total=len(df)):
                vid = imageio.get_reader(f'{vid_dir}/{vid}', 'ffmpeg')
                cur_act = []
                for i in range(90):
                    input_img = preprocess(Image.fromarray(vid.get_data(i)))
                    cur_act.append(feature_extractor.forward(input_img, layer=self.layer, combination=None))
                activation.append(np.array(cur_act).mean(axis=0))
            activation = np.array(activation)
            print(activation.shape)
            np.save(out_file, activation)
        else:
            activation = np.load(out_file)
        self.get_PCs(activation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', '-l', type=int, default=2)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    AlexNetActivations(args).run()

if __name__ == '__main__':
    main()
