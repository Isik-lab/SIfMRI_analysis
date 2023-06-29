#
import os.path
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

variance_partitioning = True
if not variance_partitioning:
    figure_number_start = 1
    surface_path = f'{figure_dir}/SurfaceStats/categories'
    category_naming = 'category'
else:
    figure_number_start = 3
    surface_path = f'{figure_dir}/SurfaceStats/categories_unique'
    category_naming = 'dropped-categorywithnuisance'

for i, categories in enumerate([['moten', 'scene_object'],
                             ['social_primitive', 'social']]):
    figure_number = f'extra{i+figure_number_start}'
    # Open the canvas
    c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
    print(figure_number)

    x1 = 5
    y1 = canvas_height
    for i, (subj, category) in enumerate(product(range(4), categories)):
        if category:
            sid = str(subj+1).zfill(2)
            file = f"{surface_path}/sub-{sid}/unfiltered/sub-{sid}_{category_naming}-{category}_view-{view}_hemi-lh.png"
            print(file)
            if os.path.exists(file):
                add_img(c, file,
                        x1, y1, scaling_factor=scaling_factor)
                add_img(c, file.replace('hemi-lh', 'hemi-rh'),
                        x1+rh_shift, y1, scaling_factor=scaling_factor)

        if (x1+(horizontal_shift*1.5)) > canvas_width:
            x1 = 5
            y1 -= vertical_shift
        else:
            x1 += horizontal_shift

    # x1 = 5
    # y1 = canvas_height
    # i = 0
    # for category, subj in product(categories, range(4)):
    #     if category:
    #         c.drawString(x1, y1 - 10, alc[i])
    #         if i == 3:
    #             x1 += horizontal_shift
    #             y1 = canvas_height
    #         else:
    #             y1 -= vertical_shift
    #         i += 1

    c.save()
