#
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img


def hshifts(view_key, hemi_key):
    d = dict()
    d['lateral'] = {'lh': 20, 'rh': -20}
    d['medial'] = {'lh': 0, 'rh': 0}
    d['ventral'] = {'lh': 10, 'rh': -35}
    return d[view_key][hemi_key]


def vshifts(key):
    d = dict()
    d['lateral'] = 0
    d['medial'] = -55
    d['ventral'] = -82
    return d[key]


def scaling_factor(key):
    d = dict()
    d['lateral'] = 0.135
    d['medial'] = 0.135
    d['ventral'] = 0.175
    return d[key]


def rotation(view_key, hemi_key):
    d = dict()
    d['lateral'] = {'lh': 0, 'rh': 0}
    d['medial'] = {'lh': 90, 'rh': 270}
    d['ventral'] = {'lh': 0, 'rh': 0}
    return d[view_key][hemi_key]


def rotate_img_files(path, hemi):
    rotate_degree = {'lh': 90, 'rh': 270}
    import glob
    import PIL
    files = glob.glob(f'{path}/*view-ventral*{hemi}*')
    if not files:
        files = glob.glob(f'{path}/*{hemi}*ventral*')
    print(files)
    for file in files:
        im = PIL.Image.open(file)
        im = im.rotate(rotate_degree[hemi])
        im.save(file)


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
analysis = 'rois'
need_rotation = True
canvas_height_in = 4.1
figure_number = 'S2'
surface_path = f'{figure_dir}/ROIMap'

if need_rotation:
    rotate_img_files(surface_path, 'lh')
    rotate_img_files(surface_path, 'rh')
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
views = ['lateral', 'medial', 'ventral']
horizontal_shift = 240
rh_shift = 130
vertical_shift = 150
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
canvas_height = canvas_height_in*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)

# Open the canvas
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))

x1 = 5
y1 = canvas_height
for i, (subj, figure) in enumerate(zip(range(4), ['a', 'b', 'c', 'd'])):
    sid = str(subj+1).zfill(2)
    c.drawString(x1, y1 - 10, figure)
    for view in views:
        file = f"{surface_path}/sub-{sid}_roi-map_view-{view}_hemi-lh.png"
        add_img(c, file,
                x1+hshifts(view, 'lh'), y1+vshifts(view),
                scaling_factor=scaling_factor(view), rotate=rotation(view, 'lh'))
        add_img(c, file.replace('hemi-lh', 'hemi-rh'),
                x1+hshifts(view, 'rh')+rh_shift, y1+vshifts(view),
                scaling_factor=scaling_factor(view), rotate=rotation(view, 'rh'))
    if (x1+(horizontal_shift*1.5)) > canvas_width:
        x1 = 5
        y1 -= vertical_shift
    else:
        x1 += horizontal_shift

c.save()
