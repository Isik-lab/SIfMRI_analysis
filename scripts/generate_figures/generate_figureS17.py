import itertools
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_img
import string

process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
canvas_height_in = 2.25
figure_number = 'S17'
surface_path = f'{figure_dir}/SurfaceStats/features_unique'
features = ['transitivity', 'communication']
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
view = 'lateral'
horizontal_shift = 250
rh_shift = 100
vertical_shift = 85
scaling_factor = 0.15
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
for i, (subj, feature) in enumerate(itertools.product(range(4), features)):
    sid = str(subj + 1).zfill(2)
    figure = alphabet[i]
    file = f"{surface_path}/sub-{sid}/sub-{sid}_dropped-featurewithnuisance-{feature}_view-{view}_hemi-lh.png"
    if os.path.exists(file):
        print(file)
        add_img(c, file,
                x1, y1,
                scaling_factor=scaling_factor)
        add_img(c, file.replace('hemi-lh', 'hemi-rh'),
                x1+rh_shift, y1,
                scaling_factor=scaling_factor)
        c.drawString(x1, y1-10, figure)
        if (x1+(horizontal_shift*1.5)) > canvas_width:
            x1 = 5
            y1 -= vertical_shift
        else:
            x1 += horizontal_shift

c.rotate(90)
c.setFont("Helvetica", 5)
c.drawString(canvas_height-65, -215, "Explained variance (r2)")
c.save()
