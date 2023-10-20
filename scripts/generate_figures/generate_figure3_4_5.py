from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.tools import add_svg, add_img


process = 'PaperFigures'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
canvas_height_in = 5
s_num = 2
sid = str(s_num).zfill(2)
hemis = ['lh', 'rh']
views = ['lateral']#, 'medial', 'ventral']
rh_shift = 195
scaling_factor = .3
canvas_width, _ = letter
pixel_per_in = canvas_width/8.5
print(pixel_per_in)
canvas_height = (canvas_height_in)*pixel_per_in
margins = 2 #inches
canvas_width = canvas_width - (pixel_per_in * margins)
print(canvas_width, canvas_height)

for analysis in ['categories', 'categories_unique', 'features_unique']:
    if analysis == 'categories':
        figure_number = 3
        plot_name = 'category'
        surface_path = f'{figure_dir}/PrefMap'
    elif analysis == 'categories_unique':
        figure_number = 4
        plot_name = 'dropped-categorywithnuisance'
        surface_path = f'{figure_dir}/PrefMap'
    elif analysis == 'features_unique':
        figure_number = 5
        plot_name = 'dropped-featurewithnuisance'
        surface_path = f'{figure_dir}/SurfaceStats/features_unique/sub-{sid}/filtered'
    else:
        raise Exception('analysis input must be categories categories_unique, or features_unique')
    barplot_file = f'{figure_dir}/PlotROIPrediction/group_lateral-rois_{plot_name}.svg'
    out_path = f'{figure_dir}/{process}'
    Path(out_path).mkdir(exist_ok=True, parents=True)
    print(analysis)
    print(surface_path)

    # Add the barplot
    x1 = 0
    c = canvas.Canvas(f'{out_path}/figure{figure_number}.pdf', pagesize=(canvas_width, canvas_height))
    y_pos, _ = add_svg(c, barplot_file, x1, canvas_height)
    c.setFont("Helvetica", 10)
    c.drawString(x1+5, canvas_height-10, 'a')

    y1 = y_pos + 5
    x2 = x1 - 15
    c.drawString(x1+5, y1-25, 'b')
    for view in views:
        if analysis == 'categories':
            file = f"{surface_path}/sub-{sid}_category_preference_filtered_view-{view}_hemi-lh.png"
        elif analysis == 'categories_unique':
            file = f"{surface_path}/sub-{sid}_uniquecategory_preference_filtered_view-{view}_hemi-lh.png"
        else:
            file = f"{surface_path}/sub-{sid}_dropped-featurewithnuisance-communication_view-{view}_hemi-lh.png"
        add_img(c, file,
                x2, y1,
                scaling_factor=scaling_factor)
        add_img(c, file.replace('hemi-lh', 'hemi-rh'),
                x2 + rh_shift, y1,
                scaling_factor=scaling_factor)

    # c.rotate(90)
    # c.setFont("Helvetica", 5)
    # c.drawString((canvas_height_add_in*pixel_per_in)+12, -215, "Explained variance (r2)")
    c.save()
