#
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from matplotlib import cm
import pandas as pd
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def str2ints(box):
    box = box.strip('][').split(', ')
    return np.array(box).astype('int').tolist()


def plt_fixation(image, heatmap, bb1, bb2, out_name,
                 bb_color=(255, 255, 255), bb_width=5,
                 cbar_max=.1):
    # Paste heatmap onto image
    image.paste(heatmap, (0, 0), heatmap)

    # Draw bounding boxes if available
    draw = ImageDraw.Draw(image)
    if bb1:
        draw.rectangle(bb1, outline=bb_color, width=bb_width)
    if bb2:
        draw.rectangle(bb2, outline=bb_color, width=bb_width)

    # Convert the PIL image to a NumPy array
    img_array = np.array(image)

    # Use Matplotlib to plot image and add colorbar
    plt.imshow(img_array, cmap="magma", vmin=0, vmax=cbar_max)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticks(list(np.linspace(0, cbar_max, 3)))
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Proportion of fixations', rotation=270, labelpad=20, fontsize=18)
    plt.savefig(out_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def heatmap2img(heatmap, outsize, alpha=100, max_proportion=0.15):
    scaled_heatmap = heatmap / max_proportion
    color_map = np.uint8(cm.magma(scaled_heatmap) * 255)
    color_map[:, :, -1] = alpha
    im = Image.fromarray(color_map)
    return im.resize(outsize, resample=Image.NEAREST)


def load_bounding_boxes(data_path, frame_size, resolution):
    def face_map(s, out_size):
        mat = np.zeros((out_size, out_size), dtype='bool')
        mat[s.top:s.bottom+1, s.left:s.right+1] = True
        mat = mat.flatten()
        for i in range(len(mat)):
            s[f'face{i}'] = mat[i]
        return s
    print('loading bounding boxes...')
    bb = pd.read_csv(f'{data_path}/face_annotation/bounding_boxes.csv')
    bb_average = bb[['video_name', 'face', 'top', 'left', 'bottom', 'right']]
    bb_average = bb_average.groupby(['video_name', 'face']).mean(numeric_only=True).reset_index(drop=False)
    bb_average[['top', 'left', 'bottom', 'right']] = bb_average[['top', 'left', 'bottom', 'right']] / (frame_size / resolution)
    bb_average[['top', 'left', 'bottom', 'right']] = bb_average[['top', 'left', 'bottom', 'right']].astype('int')
    bb_average = bb_average.apply(face_map, args=(resolution,), axis=1)
    return bb.set_index(['video_name', 'frame', 'face']),\
           bb_average.groupby('video_name').sum(numeric_only=False).reset_index(drop=False)


class PlotHeatmaps:
    def __init__(self, args):
        self.process = 'PlotHeatmaps'
        self.res = args.resolution
        self.frame_size = 500
        self.heatmap_max = .1
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        self.subjs = [4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]

    def heatmap_visualization(self, average_heatmap, bb):
        import cv2, mmcv
        def load_frame(path, video, frame_num=0):
            in_file = f'{path}/videos/{video}'
            video_obj = mmcv.VideoReader(in_file)
            frame = video_obj[frame_num]
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.putalpha(255)
            return frame

        heatmap_cols = [f'heatmap{i}' for i in range(self.res**2)]
        print('saving heatmap visualization...')
        for _, row in tqdm(average_heatmap.iterrows(), total=len(average_heatmap)):
            heatmap = row[heatmap_cols].to_numpy().astype('float').reshape((self.res, self.res))
            if not np.all(np.isnan(heatmap)):
                frame = load_frame(self.data_dir, row.video_name)
                file = f"{self.figure_dir}/{row.video_name.replace('mp4', 'png')}"
                heatmap_img = heatmap2img(heatmap, frame.size, max_proportion=self.heatmap_max)

                if (row.video_name, 1, 'face1') in bb.index:
                    bb1 = str2ints(bb.loc[row.video_name, 1, 'face1'].box)
                else:
                    bb1 = None

                if (row.video_name, 1, 'face2') in bb.index:
                    bb2 = str2ints(bb.loc[row.video_name, 1, 'face2'].box)
                else:
                    bb2 = None

                plt_fixation(frame, heatmap_img, bb1, bb2, file, cbar_max=self.heatmap_max)
            else:
                print(f'no data for {row.video_name}')

    def load_heatmaps(self):
        df =[]
        print('loading heatmaps...')
        for subj in tqdm(self.subjs):
            subj_str = str(subj).zfill(3)
            subj_heatmap = pd.read_csv(f'{self.out_dir}/EyeTracking_WithinSubj/heatmaps/subj{subj_str}.csv')
            subj_avg_heatmap = subj_heatmap.groupby(['video_name']).mean(numeric_only=True).reset_index()
            subj_avg_heatmap['subj'] = f'subj{subj_str}'
            df.append(subj_avg_heatmap)
        return pd.concat(df).groupby('video_name').mean(numeric_only=True).reset_index()

    def run(self):
        df = self.load_heatmaps()
        bounding_boxes, _ = load_bounding_boxes(self.data_dir, self.frame_size, self.res)
        self.heatmap_visualization(df, bounding_boxes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=20)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PlotHeatmaps(args).run()


if __name__ == '__main__':
    main()