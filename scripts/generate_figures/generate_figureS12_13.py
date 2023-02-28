import itertools
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_img
import string


def hshifts(view_key, hemi_key):
    d = dict()
    d['lateral'] = {'lh': 25, 'rh': -25}
    d['medial'] = {'lh': 0, 'rh': 0}
    d['ventral'] = {'lh': 14, 'rh': -40}
    return d[view_key][hemi_key]


def vshifts(key):
    d = dict()
    d['lateral'] = 0
    d['medial'] = -40
    d['ventral'] = -55
    return d[key]


def scaling_factor(key):
    d = dict()
    d['lateral'] = 0.1
    d['medial'] = 0.1
    d['ventral'] = 0.14
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
canvas_height_in = 7
surface_path = f'{figure_dir}/SurfaceStats/features_unique'
# features = ['transitivity', 'communication']
feature = 'facingness'
need_rotation = True
if feature == 'transitivity':
    figure_number = 'S12'
elif feature == 'communication':
    figure_number = 'S13'
elif feature == 'facingness':
    figure_number = 'S14'
else:
    raise Exception('feature input must be communication, facingness, or transitivity')
if need_rotation:
    rotate_img_files(f'{surface_path}/sub*/*/', 'lh')
    rotate_img_files(f'{surface_path}/sub*/*/', 'rh')
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
views = ['lateral', 'medial', 'ventral']
filtering = ['filtered', 'unfiltered']
horizontal_shift = 250
rh_shift = 120
vertical_shift = 125
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
canvas_height = canvas_height_in*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
alphabet = string.ascii_lowercase

# Open the canvas
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))

x1 = 5
y1 = canvas_height
for i, (subj, filter) in enumerate(itertools.product(range(4), filtering)):
    sid = str(subj + 1).zfill(2)
    figure = alphabet[i]
    c.drawString(x1, y1 - 10, figure)
    for view in views:
        file = f"{surface_path}/sub-{sid}/{filter}/sub-{sid}_dropped-featurewithnuisance-{feature}_view-{view}_hemi-lh.png"
        add_img(c, file,
                x1 + hshifts(view, 'lh'), y1 + vshifts(view),
                scaling_factor=scaling_factor(view), rotate=rotation(view, 'lh'))
        add_img(c, file.replace('hemi-lh', 'hemi-rh'),
                x1 + hshifts(view, 'rh') + rh_shift, y1 + vshifts(view),
                scaling_factor=scaling_factor(view), rotate=rotation(view, 'rh'))
    if (x1 + (horizontal_shift * 1.5)) > canvas_width:
        x1 = 5
        y1 -= vertical_shift
    else:
        x1 += horizontal_shift

c.rotate(90)
c.setFont("Helvetica", 5)
c.drawString(canvas_height-65, -215, "Explained variance (r2)")
c.save()
