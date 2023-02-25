import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img


def hshifts(view_key, hemi_key):
    d = dict()
    d['lateral'] = {'lh': 32, 'rh': -32}
    d['medial'] = {'lh': 0, 'rh': 0}
    d['ventral'] = {'lh': 22, 'rh': -60}
    return d[view_key][hemi_key]


def vshifts(key):
    d = dict()
    d['lateral'] = 0
    d['medial'] = -85
    d['ventral'] = -125
    return d[key]


def scaling_factor(key):
    d = dict()
    d['lateral'] = 0.2
    d['medial'] = 0.2
    d['ventral'] = 0.25
    return d[key]


def rotation(view_key, hemi_key):
    d = dict()
    d['lateral'] = {'lh': 0, 'rh': 0}
    d['medial'] = {'lh': 90, 'rh': 270}
    d['ventral'] = {'lh': 0, 'rh': 0}
    return d[view_key][hemi_key]


def rotate_img_files(path, hemi):
    rotate_degree = {'lh': 90, 'rh': 270}
    import glob
    import PIL
    files = glob.glob(f'{path}/*view-ventral*{hemi}*')
    if not files:
        files = glob.glob(f'{path}/*{hemi}*ventral*')
    print(files)
    for file in files:
        im = PIL.Image.open(file)
        im = im.rotate(rotate_degree[hemi])
        im.save(file)


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
analysis = 'categories'
need_rotation = False
canvas_height_in = 4.5
figure_number = 5
plot_name = 'dropped-featurewithnuisance'
sid = str(2).zfill(2)
surface_path = f'{figure_dir}/SurfaceStats/features_unique/sub-{sid}/filtered'
print(os.path.exists(surface_path))
if need_rotation:
    rotate_img_files(surface_path, 'lh')
    rotate_img_files(surface_path, 'rh')
barplot_file = f'{figure_dir}/PlotROIPrediction/group_lateral-rois_{plot_name}.svg'
out_path = f'{figure_dir}/{process}'
Path(out_path).mkdir(exist_ok=True, parents=True)
hemis = ['lh', 'rh']
views = ['lateral']#, 'medial', 'ventral']
horizontal_shift = 250
rh_shift = 190
verticle_shift = 85
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
print(pixel_per_in)
canvas_height = (canvas_height_in)*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
print(canvas_width, canvas_height)

# Add the barplot
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
y_pos, _ = add_svg(c, barplot_file, 1, canvas_height)
c.setFont("Helvetica", 10)
c.drawString(5, canvas_height-10, 'a')

x1 = 80
y1 = y_pos
c.drawString(x1+20, y1-20, 'b')
for view in views:
    file = f"{surface_path}/sub-{sid}_dropped-featurewithnuisance-communication_view-{view}_hemi-lh.png"
    add_img(c, file,
            x1 + hshifts(view, 'lh'), y1 + vshifts(view),
            scaling_factor=scaling_factor(view), rotate=rotation(view, 'lh'))
    add_img(c, file.replace('hemi-lh', 'hemi-rh'),
            x1 + hshifts(view, 'rh') + rh_shift, y1 + vshifts(view),
            scaling_factor=scaling_factor(view), rotate=rotation(view, 'rh'))

# c.rotate(90)
# c.setFont("Helvetica", 5)
# c.drawString((canvas_height_add_in*pixel_per_in)+12, -215, "Explained variance (r2)")
c.save()