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
analysis = 'categories'
canvas_height_in = 4
if analysis == 'categories':
    figure_number = 3
    plot_name = 'category'
    categories = ['alexnet', 'moten', 'scene_object', 'social_primitive', 'social']
    canvas_height_add_in = 1
elif analysis == 'categories_unique':
    figure_number = 4
    plot_name = 'dropped-categorywithnuisance'
    categories = ['alexnet', 'moten', 'scene_object', 'social_primitive', 'social']
    canvas_height_add_in = 1
elif analysis == 'features_unique':
    figure_number = 5
    plot_name = 'dropped-featurewithnuisance'
    categories = ['transitivity', 'communication']
    canvas_height_add_in = 0
else:
    raise Exception('analysis input must be categories, categories_unique, or feature_unique')
sid = str(2).zfill(2)
surface_path = f'{figure_dir}/SurfaceStats/{analysis}/sub-{sid}'
barplot_file = f'{figure_dir}/PlotROIPrediction/group_lateral-rois_{plot_name}.svg'
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
view = 'lateral'
horizontal_shift = 160
verticle_shift = 70
scaling_factor = 0.105
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
print(pixel_per_in)
canvas_height = (canvas_height_in+canvas_height_add_in)*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
print(canvas_width, canvas_height)

# Add the barplot
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
y_pos = add_svg(c, barplot_file, 1, canvas_height)
c.setFont("Helvetica", 10)
c.drawString(5, canvas_height-10, 'a')

x1 = 5
y1 = y_pos - 10
for i, (category, figure) in enumerate(zip(categories, ['b', 'c', 'd', 'e', 'f'])):
    print(f"{surface_path}/sub-{sid}_{plot_name}-{category}_view-{view}_hemi-lh.png")
    print(x1, y1)
    add_img(c, f"{surface_path}/sub-{sid}_{plot_name}-{category}_view-{view}_hemi-lh.png",
            x1, y1,
            scaling_factor=scaling_factor)
    add_img(c, f"{surface_path}/sub-{sid}_{plot_name}-{category}_view-{view}_hemi-rh.png",
            x1+65, y1,
            scaling_factor=scaling_factor)
    c.drawString(x1, y1, figure)
    if (x1+(horizontal_shift*1.5)) > canvas_width:
        x1 = 5
        y1 -= verticle_shift
    else:
        x1 += horizontal_shift

c.rotate(90)
c.setFont("Helvetica", 3)
c.drawString((canvas_height_add_in*pixel_per_in)+18, -147, "Explained variance (r2)")
c.save()
