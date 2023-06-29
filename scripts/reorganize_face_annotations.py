import pandas as pd
import numpy as np
import json
from tqdm import tqdm


def draw_bounding_boxes(in_file, out_file, boxes):
    import mmcv, cv2
    from PIL import Image, ImageDraw

    video_obj = mmcv.VideoReader(in_file)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video_obj]

    frames_tracked = []
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for face in ['face1', 'face2']:
            try:
                box = boxes.loc[(i + 1, face), 'box']
                draw.rectangle(box, outline=(255, 255, 255), width=2)
            except:
                print(f'no box for frame {i+1} {face}')

        # Add to frame list
        frames_tracked.append(frame_draw.resize((500, 500), Image.BILINEAR))

    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_tracked = cv2.VideoWriter(out_file, fourcc, 30.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()


def reorg_annotations(in_file, out_file):
    with open(in_file, 'r') as f:
        data = json.loads(f.read())
    f.close()

    frame_center = np.array((250., 250.))
    bounding_box_data = []
    for video_json in data:
        video_name = video_json['data_row']['external_id']
        if video_json['projects']['clit3zloh00k4071d6x1lc5ej']['labels']:
            frames_json = video_json['projects']['clit3zloh00k4071d6x1lc5ej']['labels'][0]['annotations']['frames']
            for frame in frames_json.keys():
                for face in frames_json[frame]['objects'].keys():
                    top = int(frames_json[frame]['objects'][face]['bounding_box']['top'])
                    left = int(frames_json[frame]['objects'][face]['bounding_box']['left'])
                    height = int(frames_json[frame]['objects'][face]['bounding_box']['height'])
                    width = int(frames_json[frame]['objects'][face]['bounding_box']['width'])
                    bottom = top + height
                    right = left + width
                    center = np.array((left + (width/2), top + (height/2)))
                    frame_data = {'video_name': video_name,
                                  'frame': int(frame),
                                  'face': frames_json[frame]['objects'][face]['name'],
                                  'left': left, 'top': top, 'right': right, 'bottom': bottom,
                                  'box': [left, top, right, bottom],
                                  'height': height, 'width': width,
                                  'x_center': center[0],
                                  'y_center': center[1],
                                  'face_area': height*width,
                                  'face_centrality': np.linalg.norm(frame_center - center)}
                    bounding_box_data.append(frame_data)
    df = pd.DataFrame(bounding_box_data)
    df.to_csv(out_file, index=False)
    return df


def count_remaining_videos(df):
    videos = pd.read_csv('../data/raw/annotations/video_names.txt', header=None)
    videos.columns = ['video_name']

    labeled = df.drop_duplicates('video_name')
    labeled['labeled'] = 'yes'
    labeled_videos = labeled.drop_duplicates('video_name')
    missing_videos = videos.merge(labeled_videos[['video_name', 'labeled']], how='outer')
    missing_videos = missing_videos[missing_videos.labeled.isna()]
    print(f'{len(missing_videos)} videos still to annotate')


def summarize_df(df, out_file):
    frame_average = df.groupby(['video_name', 'face']).mean().reset_index(drop=False)
    summary_df = frame_average[['video_name', 'face_area']].groupby('video_name').sum().reset_index(drop=False)
    summary_df['face_centrality'] = frame_average[['video_name', 'face_centrality']].groupby('video_name').min().reset_index(drop=False)['face_centrality']
    summary_df.to_csv(out_file, index=False)


df = reorg_annotations('../data/raw/face_annotation/annotations.json',
                       '../data/raw/face_annotation/bounding_boxes.csv')
summarize_df(df, '../data/raw/annotations/face_annotations.csv')
annotated_videos = count_remaining_videos(df)
# for video in df.video_name.unique():
#     print(video)
#     draw_bounding_boxes(f'../data/raw/videos/{video}',
#                         f'../data/interim/FaceDetection/{video}',
#                         df.loc[df.video_name == video].set_index(['frame', 'face']))


