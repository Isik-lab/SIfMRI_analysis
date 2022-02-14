# /Applications/anaconda3/envs/pytorch/bin/python

import argparse
import glob
from tqdm import tqdm
import copy

import pandas as pd
import numpy as np

import imageio
from PIL import Image

import torchvision.models as models
from torchvision import transforms as trn
from torch.autograd import Variable as V
import torch.nn as nn

def combine(X, combination=None):
    X = X.data.numpy()

    # Take the mean of the spatial axes
    if X.ndim > 2 and combination is not None:
        if combination == 'mean':
            X = X.mean(axis=(X.ndim-2, X.ndim-1))
        elif combination == 'max':
            X = X.max(axis=(X.ndim-2, X.ndim-1))
    return X.flatten()

def preprocess(image_fname, resize=224):
    center_crop = trn.Compose([
        trn.Resize((resize, resize)), #(256, 256)
        #trn.CenterCrop(crop),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if isinstance(image_fname, str):
        img_input = Image.open(image_fname)
    else:
        img_input = image_fname
    return V(center_crop(img_input).unsqueeze(0))

class alexnet_extractor(nn.Module):
    def __init__(self, net):
        super(alexnet_extractor, self).__init__()
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

class alexnet_activations():
    def __init__(self, args):
        self.process = 'alexnet_activations'
        self.layer = args.layer
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def run(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/train.csv')
        df.sort_values(by=['video_name'], inplace=True)
        vid_dir = f'{self.data_dir}/videos'

        model = models.alexnet(pretrained=True)
        model.eval()
        feature_extractor = alexnet_extractor(model)

        activation = []
        for vid in tqdm(df.video_name, total=len(df)):
            vid = imageio.get_reader(f'{vid_dir}/{vid}', 'ffmpeg')
            cur_act = []
            for i in range(90):
                input_img = preprocess(Image.fromarray(vid.get_data(i)))
                cur_act.append(feature_extractor.forward(input_img, layer=self.layer, combination=None))
            activation.append(np.array(cur_act).mean(axis=0))
        activation = np.array(activation).T
        np.save(f'{self.output_dir}/alexnet_conv{selflayer}_avgframe.npy',activation)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', '-l', type=int)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    args = parser.parse_args()
    times = alexnet_activations(args).run()

if __name__ == '__main__':
    main()
