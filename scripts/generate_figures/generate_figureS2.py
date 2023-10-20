from itertools import product
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img
from string import ascii_lowercase as alc

process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
canvas_height_in = 8
figure_number = 'S2'
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
vertical_shift = 200
horizontal_shift = 150
reliability_path = f'{figure_dir}/Reliability'
full_path = f'{figure_dir}/SurfaceStats/full'
both_paths = [reliability_path, full_path]
view = 'lateral'
rh_shift = 100

#Get width
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
#Get height in pixels
canvas_height = canvas_height_in*pixel_per_in

# Open canvas
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
x0 = 5
y0 = canvas_height - 10

# Add the lateral individual bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_lateral-rois_full-model.svg'
y_pos, scaling_factor = add_svg(c, barplot_file, x0, y0)
c.drawString(x0, y0, 'a')

# Add the ventral group bar plot
y1 = y_pos - 10
barplot_file = f'{figure_dir}/PlotROIPrediction/group_ventral-rois_full-model.svg'
print(scaling_factor)
add_svg(c, barplot_file, x0, y1,
        scaling_factor=scaling_factor)
c.drawString(x0, y1, 'b')
#
# Add the ventral individual bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_ventral-rois_full-model.svg'
print(scaling_factor)
add_svg(c, barplot_file, x0 + horizontal_shift, y1,
        scaling_factor=scaling_factor)
c.drawString(x0 + horizontal_shift, y1, 'c')

scaling_factor = 0.135
horizontal_shift = 230
rh_shift = 100
vertical_shift = 75
x1 = 5
surface_y1 = y1 - 135
for i, (subj, surface_path) in enumerate(product(range(4), both_paths)):
    sid = str(subj+1).zfill(2)
    if 'Reliability' in surface_path:
        file = f"{surface_path}/sub-{sid}_space-T1w_desc-test-fracridge_hemi-lh_view-{view}.png"
    else:
        file = f"{surface_path}/sub-{sid}/filtered/sub-{sid}_full-model_view-{view}_hemi-lh.png"
    add_img(c, file,
            x1, surface_y1, scaling_factor=scaling_factor)
    add_img(c, file.replace('hemi-lh', 'hemi-rh'),
            x1+rh_shift, surface_y1, scaling_factor=scaling_factor)
    if (x1+(horizontal_shift*1.5)) > canvas_width:
        x1 = 5
        surface_y1 -= vertical_shift
    else:
        x1 += horizontal_shift

x1 = 5
surface_y1 = y1 - 135
for i, (subj, feature) in enumerate(product(both_paths, range(4))):
    x, y = x1, surface_y1 - 10
    print(x, y)
    c.drawString(x, y, alc[i+3])
    if i == 3:
        x1 += horizontal_shift
        surface_y1 = y1 - 135
    else:
        surface_y1 -= vertical_shift

c.save()
