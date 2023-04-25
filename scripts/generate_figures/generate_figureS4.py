#
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img
from string import ascii_lowercase as alc
from itertools import product


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
analysis = 'rois'
need_rotation = True
canvas_height_in = 4.1
figure_number = 'S4'
reliability_path = f'{figure_dir}/Reliability'
full_path = f'{figure_dir}/SurfaceStats/full'
both_paths = [reliability_path, full_path]

out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
view = 'lateral'
horizontal_shift = 240
rh_shift = 100
vertical_shift = 75
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
canvas_height = canvas_height_in*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
scaling_factor = 0.135

# Open the canvas
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))

x1 = 5
y1 = canvas_height
for i, (subj, surface_path) in enumerate(product(range(4), both_paths)):
    sid = str(subj+1).zfill(2)
    if 'Reliability' in surface_path:
        file = f"{surface_path}/sub-{sid}_space-T1w_desc-test-fracridge_hemi-lh_view-{view}.png"
    else:
        file = f"{surface_path}/sub-{sid}_full-model_view-{view}_hemi-lh.png"
    add_img(c, file,
            x1, y1, scaling_factor=scaling_factor)
    add_img(c, file.replace('hemi-lh', 'hemi-rh'),
            x1+rh_shift, y1, scaling_factor=scaling_factor)
    if (x1+(horizontal_shift*1.5)) > canvas_width:
        x1 = 5
        y1 -= vertical_shift
    else:
        x1 += horizontal_shift

x1 = 5
y1 = canvas_height
for i, (subj, feature) in enumerate(product(both_paths, range(4))):
    c.drawString(x1, y1 - 10, alc[i])
    if i == 3:
        x1 += horizontal_shift
        y1 = canvas_height
    else:
        y1 -= vertical_shift

c.save()
