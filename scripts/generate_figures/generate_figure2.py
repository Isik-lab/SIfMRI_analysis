from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
analysis = 'full'
canvas_height_in = 3
figure_number = 2
plot_name = 'category'
canvas_height_add_in = 0
sid = str(2).zfill(2)
barplot_file = f'{figure_dir}/PlotROIPrediction/group_lateral-rois_full-model.svg'
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
view = 'lateral'
horizontal_shift = 250
rh_shift = 100
vertical_shift = 70
scaling_factor = 0.15
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
canvas_height = (canvas_height_in+canvas_height_add_in)*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
print(canvas_width, canvas_height)

# Add the barplot
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
y_pos, _ = add_svg(c, barplot_file, 1, canvas_height)
c.setFont("Helvetica", 10)
c.drawString(5, canvas_height-10, 'a')

# Reliability
x1 = 5
y1 = y_pos
surface_path = f'{figure_dir}/Reliability/'
print(x1, y1)
add_img(c, f"{surface_path}/sub-{sid}_space-T1w_desc-test-fracridge_hemi-lh_view-{view}.png",
        x1, y1,
        scaling_factor=scaling_factor)
add_img(c, f"{surface_path}/sub-{sid}_space-T1w_desc-test-fracridge_hemi-rh_view-{view}.png",
        x1+rh_shift, y1,
        scaling_factor=scaling_factor)
c.drawString(x1, y1-10, 'b')

# Full Model
x1 += horizontal_shift
surface_path = f'{figure_dir}/SurfaceStats/{analysis}/sub-{sid}'
add_img(c, f"{surface_path}/sub-{sid}_full-model_view-{view}_hemi-lh.png",
        x1, y1,
        scaling_factor=scaling_factor)
add_img(c, f"{surface_path}/sub-{sid}_full-model_view-{view}_hemi-rh.png",
        x1+rh_shift, y1,
        scaling_factor=scaling_factor)
c.drawString(x1, y1-10, 'c')

c.rotate(90)
c.setFont("Helvetica", 5)
c.drawString(22, -215, "Explained variance (r2)")
c.save()
