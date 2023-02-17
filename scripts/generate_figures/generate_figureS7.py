from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
canvas_height_in = 4.5
figure_number = 'S7'
sid = str(2).zfill(2)
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
horizontal_shift = 240

#Get width
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
#Get height in pixels
canvas_height = canvas_height_in*pixel_per_in
#Define figure width, 2 inches becuase that is how big these figures should be
max_width = canvas_width / 2.1

# Open canvas
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf',
                  pagesize=(canvas_width, canvas_height))
x0 = 5
y0 = canvas_height

# Add the ventral group category bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/group_ventral-rois_category.svg'
y_pos, _ = add_svg(c, barplot_file, x0, y0+75, offset=0, max_width=max_width)
c.drawString(x0, y0-10, 'a')

# Add the ventral individual category bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_ventral-rois_category.svg'
add_svg(c, barplot_file, x0+horizontal_shift, y0+75, offset=0, max_width=max_width)
c.drawString(x0+horizontal_shift, y0-10, 'b')

# Add the ventral group category_unique bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/group_ventral-rois_dropped-categorywithnuisance.svg'
y_pos1, _ = add_svg(c, barplot_file, x0, y_pos+75, offset=0, max_width=max_width)
c.drawString(x0, y_pos-10, 'c')

# Add the ventral individual category_unique bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_ventral-rois_dropped-categorywithnuisance.svg'
add_svg(c, barplot_file,
                    x0+horizontal_shift, y_pos+75, offset=0,
                    max_width=max_width)
c.drawString(x0+horizontal_shift, y_pos-10, 'd')

# Add the ventral group feature_unique bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/group_ventral-rois_dropped-featurewithnuisance.svg'
add_svg(c, barplot_file, x0, y_pos1+75, offset=0, max_width=max_width)
c.drawString(x0, y_pos1-10, 'e')

# Add the ventral individual feature_unique bar plot
barplot_file = f'{figure_dir}/PlotROIPrediction/individual_ventral-rois_dropped-featurewithnuisance.svg'
add_svg(c, barplot_file,
        x0+horizontal_shift, y_pos1+75, offset=0,
        max_width=max_width)
c.drawString(x0+horizontal_shift, y_pos1-10, 'f')

c.save()
