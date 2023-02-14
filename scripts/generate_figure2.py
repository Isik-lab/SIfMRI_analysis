from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def scale(drawing, scaling_factor):
    """
    Scale a reportlab.graphics.shapes.Drawing()
    object while maintaining the aspect ratio
    """
    scaling_x = scaling_factor
    scaling_y = scaling_factor

    drawing.width = drawing.minWidth() * scaling_x
    drawing.height = drawing.height * scaling_y
    drawing.scale(scaling_x, scaling_y)
    return drawing

path = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures/Reliability/'
file = 'sub-01_space-T1w_desc-test-fracridge_hemi-lh_view-lateral.svg'

c = canvas.Canvas('file.pdf', pagesize=letter)
width, height = letter
pixel_per_in = width/8.5 #the width of US letter paper
print(f"width: {width/8.5} pixel/in, height: {height/11} pixel/in")
drawing = svg2rlg(f"{path}/{file}")
scaled_drawing = scale(drawing, scaling_factor=0.25)
renderPDF.draw(scaled_drawing, c,
               pixel_per_in, height-pixel_per_in-scaled_drawing.height,
               showBoundary=True )
# c.drawString(50, 30, 'My SVG Image')
c.save()