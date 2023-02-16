from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def add_img(current_canvas, file, x, y, scaling_factor=0.25):
    pil_img = ImageReader(file)
    img_width, img_height = pil_img._image._size
    new_width, new_height = int(img_width * scaling_factor), int(img_height * scaling_factor)
    current_canvas.drawImage(pil_img, x, y-new_height,
                             new_width, new_height, mask="auto")

path = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures/SurfaceStats/features_unique/sub-02'
files = ['sub-02_dropped-featurewithnuisance-expanse_view-lateral_hemi-lh.png',
'sub-02_dropped-featurewithnuisance-expanse_view-lateral_hemi-rh.png']
scaling_factor = 0.25
horizontal_distance = 150

c = canvas.Canvas('file.pdf', pagesize=letter)
canvas_width, canvas_height = letter
pixel_per_in = canvas_width/8.5 #the width of US letter paper
print(f"width: {canvas_width/8.5} pixel/in, height: {canvas_height/11} pixel/in")
for i, file in enumerate(files):
    add_img(c, f"{path}/{file}",
            pixel_per_in+(horizontal_distance*i), canvas_height-pixel_per_in)
# c.drawString(50, 30, 'My SVG Image')
c.save()