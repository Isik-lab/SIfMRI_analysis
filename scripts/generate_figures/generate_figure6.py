import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img


def hshifts(view_key, hemi_key):
    d = dict()
    d['lateral'] = {'lh': 32, 'rh': -32}
    d['medial'] = {'lh': 0, 'rh': 0}
    d['ventral'] = {'lh': 22, 'rh': -60}
    return d[view_key][hemi_key]


def vshifts(key):
    d = dict()
    d['lateral'] = 0
    d['medial'] = -85
    d['ventral'] = -125
    return d[key]


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'

canvas_height_in = 5.1
canvas_width_in = 6.5
figure_number = 6
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)

letter_width, _ = letter
pixel_per_in = letter_width/8.5
canvas_width = pixel_per_in*canvas_width_in
canvas_height = pixel_per_in*canvas_height_in
print(canvas_width, canvas_height)

# Add the barplot
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
c.setFont("Helvetica", 10)

y1 = canvas_height-10
x2 = 250
_, _ = add_svg(c, f'{figure_dir}/FeatureCorrelations/face_area.svg',
        x=5, y=y1, scaling_factor=0.8)
c.drawString(5, y1, 'A')

_, _ = add_svg(c, f'{figure_dir}/FeatureCorrelations/face_centrality.svg',
        x=x2, y=y1, scaling_factor=0.8)
c.drawString(x2, y1, 'B')


y2 = canvas_height-205
c.drawString(5, y2, 'C')
file = f"{figure_dir}/PlotHeatmaps/yt-dUQG4COAK54_46.png"
add_img(c, file, 15, y2-10, scaling_factor=.21)

file = f"{figure_dir}/PlotHeatmaps/yt-0IxYqinsuz8_7.png"
add_img(c, file, 130, y2-10, scaling_factor=.21)

_, _ = add_svg(c, f'{figure_dir}/EyeTracking_BetweenSubj/face_feature.svg',
        x=x2, y=y2, scaling_factor=0.8)
c.drawString(x2, y2, 'D')

c.save()