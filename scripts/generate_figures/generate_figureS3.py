from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
canvas_height_in = 4
figure_number = 'S3'
sid = str(2).zfill(2)
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
vertical_shift = 200
horizontal_shift = 150

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
y0 = canvas_height


# Add the lateral individual bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_lateral-rois_full-model.svg'
y_pos, scaling_factor = add_svg(c, barplot_file, x0, y0)
c.drawString(x0, y0-10, 'a')

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

c.save()
