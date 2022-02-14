import argparse
import imageio
import moten
import numpy as np
import pandas as pd

class of_activations():
    def __init__(self, args):
        self.process = 'of_activations'
        self.layer = args.layer
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def run(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/train.csv')
        df.sort_values(by=['video_name'], inplace=True)
        vid_dir = f'{self.data_dir}/videos'

        # Create a pyramid of spatio-temporal gabor filters
        vdimm, hdim, fps = 500, 500, 30
        pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=fps)

        moten = []
        for vid in df.video_name:
            vid_obj = imageio.get_reader(f'{vid_dir}/{vid}', 'ffmpeg')
            vid = []
            for i in range(90):
                vid.append(vid_obj.get_data(i).mean(axis=-1))
            vid = np.array(vid)
            moten_features = pyramid.project_stimulus(vid)
            moten.append(moten_features.mean(axis=0))
        moten = np.array(moten)
        np.save(f'{out_dir}/of_activations.npy', moten)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', '-l', type=int)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    args = parser.parse_args()
    times = of_activations(args).run()

if __name__ == '__main__':
    main()
