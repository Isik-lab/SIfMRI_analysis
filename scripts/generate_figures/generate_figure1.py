from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img
from reportlab.graphics import shapes
from reportlab.lib import colors

process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
out_path = f'{figure_dir}/{process}'
figure_number = 1

canvas_height_in = 3.75
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
canvas_height = canvas_height_in*pixel_per_in
margin_inches = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margin_inches)
print(canvas_width, canvas_height)
print((canvas_width/pixel_per_in), canvas_height_in)

c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))


#Adding the description
y1 = canvas_height - 15
x0 = 10

c.setFont("Helvetica", 10)
c.drawString(x0, y1+5, 'a')

# Adding all the figures
x1 = pixel_per_in*2.5

# Add b
img_file = f'{figure_dir}/ExampleRatings/yt-KZqqB7yoVYw_11.jpg'
add_img(c, img_file, x1, y1, scaling_factor=0.0246)
barplot_file = f'{figure_dir}/ExampleRatings/yt-KZqqB7yoVYw_11.svg'
_, _ = add_svg(c, barplot_file, x1+60, y1+20, scaling_factor=0.5)
c.setFont("Helvetica", 10)
c.drawString(x1-7, y1+5, 'b')


# Add c
x2 = x1 + 150
img_file = f'{figure_dir}/ExampleRatings/flickr-4-9-9-9-8-3-1-6-25549998316_29.jpg'
add_img(c, img_file, x2, y1, scaling_factor=0.0230625)
barplot_file = f'{figure_dir}/ExampleRatings/flickr-4-9-9-9-8-3-1-6-25549998316_29.svg'
y_pos, _ = add_svg(c, barplot_file, x2+60, y1+20, scaling_factor=0.5)
c.setFont("Helvetica", 10)
c.drawString(x2-7, y1+5, 'c')

# Add feature variance

barplot_file = f'{figure_dir}/FeatureVariance/set-both.svg'
_, _ = add_svg(c, barplot_file, x1, y1+65, scaling_factor=0.45)
c.setFont("Helvetica", 10)
c.drawString(x1-7, y_pos-20, 'd')

# Add the confusion matrix
x3 = x1 + 120
barplot_file = f'{figure_dir}/FeatureRegression/feature_regression.svg'
_, _ = add_svg(c, barplot_file, x3, y1-10, scaling_factor=0.525)
c.setFont("Helvetica", 10)
c.drawString(x3-7, y_pos-20, 'e')

c.save()