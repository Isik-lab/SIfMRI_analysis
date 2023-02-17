from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
analysis = 'features_unique'
canvas_height_in = 4.1
if analysis == 'categories':
    figure_number = 3
    plot_name = 'category'
    categories = ['alexnet', 'moten', 'scene_object', 'social_primitive', 'social']
    canvas_height_add_in = 2.5
elif analysis == 'categories_unique':
    figure_number = 4
    plot_name = 'dropped-categorywithnuisance'
    categories = ['alexnet', 'moten', 'scene_object', 'social_primitive', 'social']
    canvas_height_add_in = 2.5
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
horizontal_shift = 250
rh_shift = 100
verticle_shift = 85
scaling_factor = 0.15
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
print(pixel_per_in)
canvas_height = (canvas_height_in+canvas_height_add_in)*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
print(canvas_width, canvas_height)

# Add the barplot
c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
y_pos, _ = add_svg(c, barplot_file, 1, canvas_height)
c.setFont("Helvetica", 10)
c.drawString(5, canvas_height-10, 'a')

x1 = 5
y1 = y_pos
for i, (category, figure) in enumerate(zip(categories, ['b', 'c', 'd', 'e', 'f'])):
    print(f"{surface_path}/sub-{sid}_{plot_name}-{category}_view-{view}_hemi-lh.png")
    print(x1, y1)
    add_img(c, f"{surface_path}/sub-{sid}_{plot_name}-{category}_view-{view}_hemi-lh.png",
            x1, y1,
            scaling_factor=scaling_factor)
    add_img(c, f"{surface_path}/sub-{sid}_{plot_name}-{category}_view-{view}_hemi-rh.png",
            x1+rh_shift, y1,
            scaling_factor=scaling_factor)
    c.drawString(x1, y1-10, figure)
    if (x1+(horizontal_shift*1.5)) > canvas_width:
        x1 = 5
        y1 -= verticle_shift
    else:
        x1 += horizontal_shift

c.rotate(90)
c.setFont("Helvetica", 5)
c.drawString((canvas_height_add_in*pixel_per_in)+12, -215, "Explained variance (r2)")
c.save()
