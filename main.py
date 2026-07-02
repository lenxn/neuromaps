import numpy as np
import numpy.typing as npt
import neuromaps as nm
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict
from os import path
from cmcrameri import cm
import nilearn
from enum import StrEnum, IntEnum
import os
import nilearn.datasets
import tempfile
from nilearn.plotting import plot_stat_map, plot_glass_brain


# z-loadings from the study by Kohn et al. 2021
PATH_TO_ACTIVITIES = "../neuromaps-data/loadings/"

# slight variation of the altas from (https://neurovault.org/images/1699/)
# tiny fragments have been merged with larger ones
ATLAS_FILE = "myatlas.nii"

# fixed labels for regions of the given atlas
LABELS = {
    1: 'Frontal Pole',
    2: 'Insular Cortex',
    3: 'Superior Frontal Gyrus',
    4: 'Middle Frontal Gyrus',
    5: 'Inferior Frontal Gyrus',
    6: 'Precentral Gyrus',
    7: 'Temporal Pole',
    8: 'Superior Temporal Gyrus',
    9: 'Middle Temporal Gyrus',
    10: 'Inferior Temporal Gyrus',
    11: 'Postcentral Gyrus',
    12: 'Superior Parietal Lobule',
    13: 'Supramarginal Gyrus',
    14: 'Angular Gyrus',
    15: 'Lateral Occipital Cortex',
    16: 'Intracalcarine Cortex',
    17: 'Frontal Medial Cortex',
    18: 'Juxtapositional Lobule Cortex',
    19: 'Subcallosal Cortex',
    20: 'Paracingulate Gyrus',
    21: 'Cingulate Gyrus',
    22: 'Precuneous Cortex',
    23: 'Cuneal Cortex',
    24: 'Frontal Orbital Cortex',
    25: 'Parahippocampal Gyrus',
    26: 'Lingual Gyrus',
    27: 'Temporal Fusiform Cortex',
    28: 'Temporal Occipital Fusiform Cortex',
    29: 'Occipital Fusiform Gyrus',
    30: 'Frontal Operculum Cortex',
    31: 'Central Opercular Cortex',
    32: 'Parietal Operculum Cortex',
    33: 'Planum Polare',
    34: "Heschl's Gyrus",
    35: 'Planum Temporale',
    36: 'Supracalcarine Cortex',
    37: 'Occipital Pole'
}

# line thickness for contour outlines in the saliency map
LINE_THICKNESS = 2

# qualitative color palettes for the contour outlines in the saliency map
CONTOUR_COLORS_LIGHT = [(np.array(c)/255).tolist() for c in (
    (100,186,170), (162,22,54), (36,118,114), (236,99,121), (67,57,59), (250,181,181)
)]
CONTOUR_COLORS_DARK = [(np.array(c)/255).tolist() for c in (
    (30,255,6), (228,254,0), (245,49,118), (76,252,225), (236,75,24), (100,212,253)
)]

# physical resolution of the saliency map in the background
NUM_BINS = 40
SCALE_FACTOR = 4

# arbitrary scaling factor for scalling the compas axes
COMPASS_SCALE = 13

# the treshold for the z-loading for masking active regions
ACTIVATION_THRESHOLD = 10

OUTPUT_DIR = "outputs"

class ColorMode(StrEnum):
    """
    The currently supported color modes.
    """
    LIGHT = "light"
    DARK = "dark"

class ReferenceVisualization(StrEnum):
    """
    The currently supported reference visualizations.
    """
    STAT_MAP = "stat_map"
    GLASS_BRAIN = "glass_brain"

class Coords(IntEnum):
    """
    The cartesisan coordinates of the 3D space.
    """
    X = 0
    Y = 1
    Z = 2

def _draw_neuromap(atlas: list[int],
                   loading: list[float],
                   mask: list[bool],
                   shape: tuple[int],
                   output_file_name: str,
                   beta: float=0,
                   legend_file_name:str="",
                   color_mode: ColorMode=ColorMode.LIGHT):

    saliency, labels_contours, proj_vec = nm.get_neuromap(atlas,
                    loading, shape,
                    morph_beta=beta, 
                    activity_threshold=10, 
                    projection_connectivity=nm.Projection.ConnectivityType.LABEL)

    max_val = np.max(np.abs(saliency))

    fig, ax = plt.subplots()
    if color_mode == ColorMode.DARK: ax.set_facecolor('black')
    saliency = np.transpose(cv2.resize(np.transpose(saliency, (2,1,0)), dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR), (2,1,0))
    im = ax.imshow(saliency[0], alpha=saliency[1], cmap=cm.vik if color_mode == ColorMode.DARK else cm.managua)
    im.set_clim(-max_val, max_val)
    if color_mode != ColorMode.DARK: im.set_cmap(im.get_cmap().reversed())

    def get_colors_qualitative(idx: int, color_mode: ColorMode) -> list[tuple[int, int, int]]:
        match(color_mode):
            case ColorMode.LIGHT:
                return CONTOUR_COLORS_LIGHT[idx%len(CONTOUR_COLORS_LIGHT)]
            case ColorMode.DARK:
                return CONTOUR_COLORS_DARK[idx%len(CONTOUR_COLORS_DARK)]

    i = 0
    for l,label_contours in labels_contours.items():
        if l>0:
            num_labels = len(np.array(np.where((atlas==l) & mask)).T)
            if num_labels<200: continue
            linestyle='-' if i < 6 else '--'
            for contour in label_contours:
                norm_contour = contour * SCALE_FACTOR * NUM_BINS
                ax.fill(norm_contour[:,Coords.X], norm_contour[:,Coords.Y],
                        edgecolor=get_colors_qualitative(i, color_mode),
                        linewidth=LINE_THICKNESS,
                        linestyle=linestyle, fill=False, label=LABELS[l], alpha=.5)
            i += 1
        else:
            for contour in label_contours:
                norm_brain_contour = contour * NUM_BINS * SCALE_FACTOR
                ax.fill(norm_brain_contour[:,Coords.X], norm_brain_contour[:,Coords.Y], edgecolor=(.8, .8, .8) if color_mode == ColorMode.DARK else (.5, .5, .5), linewidth=LINE_THICKNESS, fill=False, alpha=.5)

    # DRAW COMPASS
    arrow_scale = 13
    x_vec = np.array([1, 0, 0])
    y_vec = np.array([0, 1, 0])
    z_vec = np.array([0, 0, 1])
    x_vec_proj = x_vec@proj_vec.T * arrow_scale
    y_vec_proj = y_vec@proj_vec.T * arrow_scale
    z_vec_proj = z_vec@proj_vec.T * arrow_scale
    offset = np.array([12, NUM_BINS*SCALE_FACTOR - 12])

    def draw_arrow(ax, label, xy, dxy, color, arrowstyle: str='->'):
        ax.annotate(label, xy=(xy[0], xy[1]), xytext=(xy[0]+dxy[0], xy[1]+dxy[1]), 
                    arrowprops=None, color=color, ha="left",va="bottom")
        ax.annotate("", xy=(xy[0]+dxy[0], xy[1]+dxy[1]), xytext=(xy[0], xy[1]),
            arrowprops=dict(arrowstyle=arrowstyle, color=color))

    def is_facing_plot(vec: npt.NDArray[np.float32]) -> bool:
        """Returns whether the given vector is facing the plot. I.e.,
        for the plot in the lower left corner, whether its in the 
        arg range of [-pi/4, 3/4 pi]."""
        return -np.pi/4 <= np.atan2(-vec[1], vec[0]) < 3/4*np.pi

    if np.linalg.norm(x_vec_proj)/arrow_scale > 0.5:
        if is_facing_plot(x_vec_proj):
            draw_arrow(ax, "R", offset, x_vec_proj, "red")
        else:
            draw_arrow(ax, "L", offset, -x_vec_proj, "red", '-')
    if np.linalg.norm(y_vec_proj)/arrow_scale > 0.5:
        if is_facing_plot(y_vec_proj):
            draw_arrow(ax, "A", offset, y_vec_proj, "blue")
        else:
            draw_arrow(ax, "P", offset, -y_vec_proj, "blue", '-')
    if np.linalg.norm(z_vec_proj)/arrow_scale > 0.5:
        if is_facing_plot(z_vec_proj):
            draw_arrow(ax, "S", offset, z_vec_proj, "green")
        else:
            draw_arrow(ax, "D", offset, -z_vec_proj, "green", '-')

    # AXIS DECORATION
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output_file_name, bbox_inches="tight")
    plt.close()

    if len(legend_file_name):
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicates while preserving order
        by_label = OrderedDict(zip(labels, handles))
        fig_legend, ax_legend = plt.subplots()
        legend = ax_legend.legend(by_label.values(), by_label.keys(), ncol=1, frameon=False)
        fig_legend.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.transformed(fig_legend.dpi_scale_trans.inverted())
        fig_legend.savefig(legend_file_name, bbox_inches=bbox, dpi=300)
        plt.close()


def _remove_axes_decoration(display):
    first_ax = next(iter(display.axes.values()))
    fig = first_ax.ax.figure

    # Remove figure background
    fig.patch.set_visible(False)

    # Remove axes backgrounds
    for ax in fig.axes:
        ax.patch.set_visible(False)

    # Remove spines (important!)
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

def _draw_reference(loading, shape, affine, vis: ReferenceVisualization, out_file_name: str, colormode: ColorMode=ColorMode.LIGHT):
    match colormode:
        case ColorMode.LIGHT:
            cmap = cm.managua.reversed()
        case ColorMode.DARK:
            cmap = cm.vik
    vmax = .5*np.max(np.abs(loading))
    template = nilearn.datasets.load_mni152_template(resolution=1)
    match vis:
        case ReferenceVisualization.STAT_MAP:
            display = plot_stat_map(nib.Nifti1Image(loading.reshape(shape), affine=affine),
                                    bg_img=template, black_bg=False,
                                    threshold=7, cut_coords=[5, -26, 21], 
                                    draw_cross=False, annotate=False, colorbar=False,
                                    cmap=cmap, vmin=-vmax, vmax=vmax)
        case ReferenceVisualization.GLASS_BRAIN:
            display = plot_glass_brain(nib.Nifti1Image(loading.reshape(shape), affine=affine), threshold=7, cut_coords=[5, -26, 21], plot_abs=False, cmap=cmap, annotate=False, colorbar=False)
    _remove_axes_decoration(display)
    
    with tempfile.NamedTemporaryFile(suffix=".svg") as tmp:
        display.savefig(
            tmp.name,
            bbox_inches="tight",
            pad_inches=0.0,
            dpi=600,
        )
        os.system(f"inkscape {tmp.name} --export-area-drawing --export-filename={out_file_name}")
    plt.close()


def main():

    # LOAD THE DATA
    epi_data = []
    for modality in range(3):
        raw_img = nib.load(f"{PATH_TO_ACTIVITIES}/niftiOut_mi{modality+2}.nii")
        affine = raw_img.affine
        epi_data += [raw_img.get_fdata()]
    epi_data = np.array(epi_data)
    epi_data = np.moveaxis(epi_data, [4], [1])
    img_size = epi_data.shape[2]*epi_data.shape[3]*epi_data.shape[4]
    data_shape = epi_data[0,0].shape
    loadings: npt.NDArray[np.float32] = np.reshape(epi_data, (epi_data.shape[0], epi_data.shape[1], img_size))
    masks: npt.NDArray[np.int32] = np.abs(loadings) > ACTIVATION_THRESHOLD

    atlas = nib.load(ATLAS_FILE).get_fdata().astype(int)
    # the atlas is downsampled as it has double the resolution of the epi data, and flattened to a 1D array for easier processing.
    atlas = atlas[::2,::2,::2].flatten()


    # DRAW THE NEURMAPS SHOWN IN THE PAPER

    # selected modality-component pairs to draw the neuromaps for
    SAMPLES = [(0,24), (2,5), (0,3), (2,17), (2,2), (1,22), (2,0), (0,13), (0,1)]

    # selected beta values to draw the neuromaps for
    BETAS = [0, .2, .5]

    for sample in SAMPLES:
        mod = sample[0]
        comp = sample[1]
        loading = loadings[mod,comp]
        mask = masks[mod,comp]
        for color_mode in ColorMode:
            _draw_reference(loading, data_shape, affine, ReferenceVisualization.STAT_MAP, path.join(OUTPUT_DIR, f"{mod}-{comp}-stat-{color_mode.name.lower()}.pdf"), colormode=color_mode)
            for beta in BETAS:
                _draw_neuromap(atlas, loading, 
                               mask, data_shape, 
                               path.join(OUTPUT_DIR, f"{mod}-{comp}-neuromap-b{beta}-{color_mode.name.lower()}.pdf"), 
                               beta, color_mode=color_mode, 
                               legend_file_name=path.join(OUTPUT_DIR, f"{mod}-{comp}-legend-{color_mode.name.lower()}.pdf"))

if __name__ == '__main__':
    main()
