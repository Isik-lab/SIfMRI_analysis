import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def add_img(current_canvas, file, x, y, scaling_factor=0.25):
    pil_img = ImageReader(file)
    img_width, img_height = pil_img._image._size
    new_width, new_height = int(img_width * scaling_factor), int(img_height * scaling_factor)
    current_canvas.drawImage(pil_img, x, y-new_height,
                             new_width, new_height, mask="auto")


def scale_svg(drawing, scaling_factor=None, max_width=468):
    """
    Scale a reportlab.graphics.shapes.Drawing()
    object while maintaining the aspect ratio
    """
    print(drawing.width, drawing.height)
    if scaling_factor is None:
        scaling_factor = canvas_width/drawing.width
    drawing.scale(scaling_factor, scaling_factor)
    return drawing


def add_svg(current_canvas, file, x, y, offset=50):
    drawing = svg2rlg(file)
    scaled_drawing = scale_svg(drawing)
    y_pos = y-scaled_drawing.height+offset
    renderPDF.draw(scaled_drawing, current_canvas,
                   x, y_pos,
                   showBoundary=False)
    return y_pos


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
analysis = 'category'
sid = str(2).zfill(2)
surface_path = f'{figure_dir}/SurfaceStats/categories/sub-{sid}'
barplot_file = f'{figure_dir}/PlotROIPrediction/group_lateral-rois_{analysis}.svg'
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
categories = ['alexnet', 'moten', 'scene_object', 'social_primitive', 'social']
hemis = ['lh', 'rh']
view = 'lateral'
horizontal_shift = 215
scaling_factor = 0.25
canvas_width, canvas_height = letter
pixel_per_in = canvas_width/8.5
margins = 2 #inches
canvas_width, canvas_height = canvas_width - (pixel_per_in * margins), canvas_height - (pixel_per_in * margins)
print(canvas_width, canvas_height)

c = canvas.Canvas(f'{out_path}/figure3.pdf', pagesize=(canvas_width, canvas_height))
# Add the barplot
y_pos = add_svg(c, barplot_file, 1, canvas_height)
c.drawString(5, canvas_height-10, 'a')

x1 = 5
y1 = y_pos
for i, (category, figure) in enumerate(zip(categories, ['b', 'c', 'd', 'e', 'f'])):
    print(f"{surface_path}/sub-{sid}_category-{category}_view-{view}_hemi-lh.png")
    print(x1, y1)
    add_img(c, f"{surface_path}/sub-{sid}_category-{category}_view-{view}_hemi-lh.png",
            x1, y1,
            scaling_factor=0.15)
    add_img(c, f"{surface_path}/sub-{sid}_category-{category}_view-{view}_hemi-rh.png",
            x1+100, y1,
            scaling_factor=0.15)
    c.drawString(x1, y1, figure)
    if (x1+(horizontal_shift*2)) > canvas_width:
        x1 = 5
        y1 -= 100
    else:
        x1 += horizontal_shift

c.save()
