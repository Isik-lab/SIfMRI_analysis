from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
analysis = 'full_model'
canvas_height_in = 2.25
if analysis == 'reliability':
    figure_number = 'S4'
    surface_path = f'{figure_dir}/Reliability'
elif analysis == 'full_model':
    figure_number = 'S5'
    surface_path = f'{figure_dir}/SurfaceStats/full'
else:
    raise Exception('analysis input must be reliability or full_model')
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
view = 'lateral'
horizontal_shift = 250
rh_shift = 100
verticle_shift = 85
scaling_factor = 0.15
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
    if analysis == 'reliability':
        file = f"{surface_path}/sub-{sid}_space-T1w_desc-test-fracridge_hemi-lh_view-{view}.png"
    else:
        file = f"{surface_path}/sub-{sid}/sub-{sid}_full-model_view-{view}_hemi-lh.png"
    add_img(c, file,
            x1, y1,
            scaling_factor=scaling_factor)
    add_img(c, file.replace('hemi-lh', 'hemi-rh'),
            x1+rh_shift, y1,
            scaling_factor=scaling_factor)
    c.drawString(x1, y1-10, figure)
    if (x1+(horizontal_shift*1.5)) > canvas_width:
        x1 = 5
        y1 -= verticle_shift
    else:
        x1 += horizontal_shift

c.rotate(90)
c.setFont("Helvetica", 5)
c.drawString(canvas_height-65, -215, "Explained variance (r2)")
c.save()
