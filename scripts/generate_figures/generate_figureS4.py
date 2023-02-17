from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
canvas_height_in = 9
figure_number = 'S4'
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
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf',
                  pagesize=(canvas_width, canvas_height))
x0 = 5
y0 = canvas_height

# Add the lateral individual bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_lateral-rois_category.svg'
y_pos, _ = add_svg(c, barplot_file, x0, y0+10)
print(y_pos)
c.drawString(x0, y0-10, 'a')

# Add the ventral group bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_lateral-rois_dropped-categorywithnuisance.svg'
y_pos1, _ = add_svg(c, barplot_file, x0, y_pos)
print(y_pos)
c.drawString(x0, y_pos-20, 'b')

# Add the ventral individual bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_lateral-rois_dropped-featurewithnuisance.svg'
add_svg(c, barplot_file, x0, y_pos1)
c.drawString(x0, y_pos1-20, 'c')

c.save()

