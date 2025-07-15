# NB need to use environment with python3.9 or above for ccf_streamlines to run
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import ccf_streamlines.projection as ccfproj
import pathlib
import pandas as pd
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import LogNorm
import copy
import seaborn as sb
import nrrd
from bg_atlasapi import BrainGlobeAtlas
import loading_functions as lf
import ast
import yaml
from scipy.ndimage import zoom
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from matplotlib.gridspec import GridSpec
from pathlib import Path
from matplotlib.ticker import LogLocator
import figure_formatting as ff
from matplotlib.patches import Patch


def make_dicts_of_mouse_3d_sample_rois(gen_parameters):
    proj_path = gen_parameters["proj_path"]
    roi_dict = {}
    sample_vol_and_regions = {}
    parameters_dict = {}
    barcodes_dict = {}
    all_bcs_with_source = {}
    barcodes_no_source = {}
    sample_vol_and_regions = {}
    scaling_factor = (
        25 / 10
    )  # since we need in 10um resolution, we need to convert 25um resolution registered cubelets to 10um
    zoom_factors = (scaling_factor, scaling_factor, scaling_factor)
    mice = gen_parameters["MICE"]
    for mouse in mice:
        lcm_directory = pathlib.Path(f"{proj_path}/{mouse}/LCM")
        # load datasets
        barcodes_across_sample = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded_with_source.pkl"
        )
        barcodes_no_source[mouse] = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded.pkl"
        )
        all_bcs_with_source[mouse] = barcodes_across_sample
        barcodes_dict[mouse] = barcodes_across_sample[
            barcodes_across_sample.sum(axis=1) > 0
        ]
        ROI_3D = np.load(lcm_directory / "ROI_3D_25.npy")
        not_in = [
            x
            for x in np.unique(ROI_3D)
            if x not in barcodes_across_sample.columns and x != 0
        ]
        ROI_3D[np.isin(ROI_3D, not_in)] = 0
        sample_vol_and_regions[mouse] = pd.read_pickle(
            lcm_directory / "sample_vol_and_regions.pkl"
        )
        parameters_dict[mouse] = lf.load_parameters(
            directory=f"{proj_path}/{mouse}/Sequencing/"
        )
        mask = np.isin(ROI_3D, parameters_dict[mouse]["cortical_samples"])
        ROI_3D[~mask] = 0
        roi_dict[mouse] = zoom(ROI_3D, zoom=zoom_factors, order=0)
        sample_vol_and_regions[mouse] = pd.read_pickle(
            lcm_directory / "sample_vol_and_regions.pkl"
        )
    return (
        roi_dict,
        barcodes_dict,
        all_bcs_with_source,
        barcodes_no_source,
        sample_vol_and_regions,
    )


def set_up_for_flatmaps(proj_path):
    convert_to_flat_path = pathlib.Path(f"{proj_path}/additional_rq")
    annotation_data = nrrd.read(f"{proj_path}/additional_rq/flatmap_butterfly.nrrd")
    labels_df = pd.read_csv(
        f"{proj_path}/additional_rq/labelDescription_ITKSNAPColor.txt",
        header=None,
        sep="\s+",
        index_col=0,
    )
    labels_df.columns = ["r", "g", "b", "x0", "x1", "x2", "acronym"]
    bf_boundary_finder = ccfproj.BoundaryFinder(
        projected_atlas_file=convert_to_flat_path / "flatmap_butterfly.nrrd",
        labels_file=convert_to_flat_path / "labelDescription_ITKSNAPColor.txt",
    )
    bf_left_boundaries = bf_boundary_finder.region_boundaries()
    bf_right_boundaries = bf_boundary_finder.region_boundaries(
        hemisphere="right_for_both",
        view_space_for_other_hemisphere="flatmap_butterfly",
    )
    proj_top = ccfproj.Isocortex2dProjector(
        convert_to_flat_path / "flatmap_butterfly.h5",
        convert_to_flat_path / "surface_paths_10_v3.h5",
        hemisphere="both",
        view_space_for_other_hemisphere="flatmap_butterfly",
    )
    ref_anno = nrrd.read(f"{proj_path}/additional_rq/annotation_25.nrrd")
    allen_anno = np.array(ref_anno)
    annotation = allen_anno[0]
    return proj_top, bf_left_boundaries, bf_right_boundaries


def generate_flatmap_dict(proj_top, roi_dict, barcodes_dict, mice):
    all_mice_flat = {}
    for i, mouse in enumerate(mice):
        roi_to_look = roi_dict[mouse]
        all_flatmap = proj_top.project_volume(roi_to_look)
        rebuilding_all_flatmap = np.zeros_like(all_flatmap)
        barcodes_across_sample = barcodes_dict[mouse]
        for roi in barcodes_across_sample.columns:
            binary_roi_array = np.where(roi_to_look == roi, roi, 0)
            flat_projection = proj_top.project_volume(binary_roi_array)
            rebuilding_all_flatmap[rebuilding_all_flatmap == 0] = flat_projection[
                rebuilding_all_flatmap == 0
            ]
        roi_labels = roi_dict[mouse]
        all_flatmap = proj_top.project_volume(roi_labels)
        barcodes_across_sample = barcodes_dict[mouse]
        barcode_matrix = np.zeros(
            (
                len(barcodes_across_sample),
                int(max(barcodes_across_sample.columns.to_list())) + 1,
            )
        )
        for column in barcodes_across_sample:
            barcode_matrix[:, int(column)] = barcodes_across_sample[column].to_numpy()
        total_counts = np.sum(barcode_matrix, axis=0)
        total_counts[0] = -1
        new_mat = np.log10(1 + total_counts[rebuilding_all_flatmap.astype(int)]).T
        new_mat = np.flip(new_mat, axis=1)
        all_mice_flat[mouse] = new_mat
    return all_mice_flat


def plot_bulk_flatmaps_indiv(
    fig,
    axes,
    proj_top,
    roi_dict,
    barcodes_dict,
    bf_left_boundaries,
    bf_right_boundaries,
    gen_parameters,
):
    mice = gen_parameters["MICE"]
    all_mice_flat = generate_flatmap_dict(
        proj_top=proj_top, roi_dict=roi_dict, barcodes_dict=barcodes_dict, mice=mice
    )
    for i, (ax, mouse) in enumerate(zip(axes[0], mice)):
        cmap = plt.cm.get_cmap("magma").copy()
        cmap.set_bad(color=[0.3, 0.3, 0.3, 1])
        im = ax.imshow(all_mice_flat[mouse], cmap=cmap)
        ax.axis("off")
        cbar = fig.colorbar(
            im, ax=ax, label="Log$_{10}$(barcode counts)", fraction=0.03, pad=0.04
        )
        cbar.set_label("Log$_{10}$(barcode counts)", rotation=270)
        for k, boundary_coords in bf_left_boundaries.items():
            ax.plot(*boundary_coords.T, c="white", lw=0.2)
        for k, boundary_coords in bf_right_boundaries.items():
            ax.plot(*boundary_coords.T, c="white", lw=0.2)
        ax.set_title(f"Distribution of MAPseq counts for {mouse}")


def plot_bulk_allen_anterograde(
    proj_top, fig, axes, bf_left_boundaries, bf_right_boundaries
):
    mcc = MouseConnectivityCache(resolution=10)
    rsp = mcc.get_reference_space()
    cortex_mask = rsp.make_structure_mask([688], direct_only=False)  # 688 = cortex ID
    expts = lf.load_allen_anterograde()
    ids = ["120491896", "116903230", "100149109"]

    for i, (ax, expt) in enumerate(zip(axes[1], expts)):
        expt_cortex = expt * cortex_mask
        projection_max = proj_top.project_volume(expt_cortex)
        to_look = np.log10(projection_max + 1).T
        to_look = np.flip(to_look, axis=1)

        im = ax.imshow(to_look, cmap="magma")
        ax.axis("off")
        fig.colorbar(
            im, ax=ax, label="Log$_{10}$(projection density)", fraction=0.03, pad=0.04
        )

        for k, boundary_coords in bf_left_boundaries.items():
            ax.plot(*boundary_coords.T, c="white", lw=0.2)
        for k, boundary_coords in bf_right_boundaries.items():
            ax.plot(*boundary_coords.T, c="white", lw=0.2)

        ax.set_title(f"Bulk GFP tracing {ids[i]}")


def get_source_2d_and_normalised_matrices(
    roi_dict, gen_parameters, proj_top, barcodes_no_source
):
    proj_path = gen_parameters["proj_path"]
    mice = gen_parameters["MICE"]
    normalised_bc_matrices = {}
    soma_coordinates_2d = []
    for mouse in mice:
        ROI_projection_max = proj_top.project_volume(roi_dict[mouse])
        # drop columns that aren't in ROI_proj_max, since these aren't cortical
        barcodes_across_sample = barcodes_no_source[mouse]
        lcm_directory = pathlib.Path(f"{proj_path}/{mouse}/LCM")
        parameters = lf.load_parameters(directory=f"{proj_path}/{mouse}/Sequencing/")
        sample_vol_and_regions = pd.read_pickle(
            lcm_directory / "sample_vol_and_regions.pkl"
        )
        sample_vol_and_regions["regions"] = sample_vol_and_regions["regions"].apply(
            ast.literal_eval
        )
        roi_list = []
        index_list = []
        for index, row in sample_vol_and_regions.iterrows():
            if any(
                "IC" in region
                or "SCs" in region
                or "SCm" in region
                or "LGd" in region
                or "LGv" in region
                or "MGv" in region
                or "RT" in region
                or "LP" in region
                or "MGd" in region
                or "P," in region
                for region in row["regions"]
            ):
                if row["ROI Number"] not in parameters["cortical_samples"]:
                    index_list.append(index)
                    roi_list.append(row["ROI Number"])
        roi_list = [
            sample for sample in roi_list if sample in barcodes_across_sample.columns
        ]
        barcodes_across_sample = barcodes_across_sample[
            barcodes_across_sample[roi_list].sum(axis=1) == 0
        ]  # we only want IT neurons

        cols_to_drop = [
            col
            for col in barcodes_across_sample.columns
            if col not in np.unique(ROI_projection_max)
        ]
        bc_matrix = barcodes_across_sample.drop(columns=cols_to_drop)
        bc_matrix = bc_matrix[
            ~(bc_matrix == 0).all(axis=1)
        ]  # drop neurons with no projections outside subcortical and source
        row_min = bc_matrix.min(axis=1)
        row_range = bc_matrix.max(axis=1) - row_min
        row_range.replace(0, np.nan, inplace=True)

        bc_matrix = bc_matrix.sub(row_min, axis=0)
        bc_mat_normalised = bc_matrix.div(row_range, axis=0)
        normalised_bc_matrices[mouse] = bc_mat_normalised
        all_bcs_with_source = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded_with_source.pkl"
        )
        all_bcs_with_source = all_bcs_with_source.loc[bc_mat_normalised.index]
        all_somas = all_bcs_with_source.idxmax(axis=1)
        soma_rois = np.unique(all_somas)
        mouse_2d = proj_top.project_volume(roi_dict[mouse])
        to_looksoma = mouse_2d.T
        to_looksoma = np.flip(to_looksoma, axis=1)
        for soma_roi in soma_rois:
            where_the_sample = np.argwhere(to_looksoma == soma_roi)
            which_2d_coord = np.mean(where_the_sample, axis=0)
            row = {
                "sample": soma_roi,
                "2d_coords_y": which_2d_coord[0],
                "2d_coords_x": which_2d_coord[1],
                "mouse": mouse,
            }
            soma_coordinates_2d.append(row)
    return normalised_bc_matrices, soma_coordinates_2d


def make_mean_3d(roi_dict, normalised_bc_matrices, gen_parameters, proj_top):
    mice = gen_parameters["MICE"]
    combined_bool_any = np.any([(arr > 0) for arr in roi_dict.values()], axis=0)
    coords = np.argwhere(combined_bool_any)
    some_mouse = mice[0]
    master_3d = np.full(roi_dict[some_mouse].shape, np.nan, dtype=float)
    roi_sum = {}
    roi_count = {}

    for m in mice:
        df = normalised_bc_matrices[m]
        col_sum = df.sum(axis=0, skipna=True).to_numpy()
        col_count = df.count(axis=0).to_numpy()
        labels = df.columns.to_numpy()
        roi_sum[m] = dict(zip(labels, col_sum))
        roi_count[m] = dict(zip(labels, col_count))

    for i, j, k in coords:
        total_sum = 0.0
        total_count = 0
        for m in mice:
            lab = roi_dict[m][i, j, k]
            if lab <= 0:
                continue
            n = roi_count[m].get(lab)
            if not n:
                continue
            total_sum += roi_sum[m][lab]
            total_count += n
        if total_count:
            master_3d[i, j, k] = total_sum / total_count

    clean_volume = np.nan_to_num(master_3d, nan=-np.inf)
    clean_volume_max = proj_top.project_volume(clean_volume)
    to_look = clean_volume_max.T
    to_look = np.flip(to_look, axis=1)
    return to_look


def plot_mean_flatmap(
    to_look, soma_coordinates_2d, fig, ax, bf_left_boundaries, bf_right_boundaries
):
    cmap = plt.cm.get_cmap("magma").copy()
    cmap.set_bad(color=[0.3, 0.3, 0.3, 1])  #
    to_look_masked = np.where(to_look <= 0, np.nan, to_look)
    im = ax.imshow(to_look_masked, cmap=cmap, norm=LogNorm())
    ax.axis("off")

    cbar = fig.colorbar(
        im, ax=ax, label="Normalised projection density", fraction=0.03, pad=0.04
    )
    cbar.set_label("Log$_{10}$(barcode counts)", rotation=270)
    for l, boundary_coords in bf_left_boundaries.items():
        ax.plot(*boundary_coords.T, c="white", lw=0.3)
    for l, boundary_coords in bf_right_boundaries.items():
        ax.plot(*boundary_coords.T, c="white", lw=0.3)
    all_soma_coords = pd.DataFrame(soma_coordinates_2d)
    coords_y = all_soma_coords["2d_coords_y"].to_numpy()
    coords_x = all_soma_coords["2d_coords_x"].to_numpy()
    ax.scatter(
        coords_x,
        coords_y,
        c="white",
        s=1.5,
        edgecolors="black",
        linewidths=0.15,
        label="Soma centroids",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_AP_position(row, dictionary):
    key = row[0]
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return None


def get_area_volumes(barcode_table_cols, lcm_directory, area_threshold=0.1):
    """
    Function to get volumes of each registered brain area from each LCM sample
    Args:
        barcode_table_cols: list of column names of the barcode matrix
        lcm_directory: path to where the lcm_directory is
        area_threshold (int): minimum value that defines a brain area within a cubelet
    Returns: area vol pandas dataframe
    """
    sample_vol_and_regions = pd.read_pickle(
        lcm_directory / "sample_vol_and_regions.pkl"
    )
    sample_vol_and_regions = sample_vol_and_regions[
        sample_vol_and_regions["ROI Number"].isin(barcode_table_cols)
    ]
    sample_vol_and_regions["regions"] = sample_vol_and_regions["regions"].apply(
        ast.literal_eval
    )
    sample_vol_and_regions["breakdown"] = sample_vol_and_regions["breakdown"].apply(
        ast.literal_eval
    )
    all_areas_unique_acronymn = np.unique(
        sample_vol_and_regions["regions"].explode().to_list()
    )
    all_area_df = pd.DataFrame(
        index=barcode_table_cols, columns=all_areas_unique_acronymn
    )
    for column in barcode_table_cols:
        index = sample_vol_and_regions[
            sample_vol_and_regions["ROI Number"] == column
        ].index
        reg = pd.DataFrame()
        reg["regions"] = [i for i in sample_vol_and_regions.loc[index, "regions"]][0]
        reg["fraction"] = [i for i in sample_vol_and_regions.loc[index, "breakdown"]][0]
        reg["vol_area"] = (
            reg["fraction"] * sample_vol_and_regions.loc[index, "Volume (um^3)"].item()
        )

        for _, row in reg.iterrows():
            all_area_df.loc[column, row["regions"]] = row["vol_area"]
    group_areas = {"Contra": all_area_df.filter(like="Contra").columns}
    areas_grouped = all_area_df.copy()
    for group, columns in group_areas.items():
        areas_grouped[group] = areas_grouped.filter(items=columns).sum(axis=1)
        columns = [value for value in columns if value in all_area_df.columns]
        areas_grouped = areas_grouped.drop(columns, axis=1)
    nontarget_list = ["fiber tracts", "root"]
    nontarget_list = [value for value in nontarget_list if value in all_area_df.columns]
    areas_only_grouped = areas_grouped.drop(nontarget_list, axis=1)
    areas_only_grouped = areas_only_grouped.apply(
        lambda row: row.where(row >= area_threshold * row.sum(), np.nan), axis=1
    )
    areas_only_grouped = areas_only_grouped.fillna(0)
    areas_only_grouped = areas_only_grouped.loc[
        :, (areas_only_grouped != 0).any(axis=0)
    ]

    return areas_only_grouped


def get_AP_coords(gen_parameters):
    mcc = MouseConnectivityCache(resolution=25)
    rsp = mcc.get_reference_space()
    VIS_mask = rsp.make_structure_mask(
        [669], direct_only=False
    )  # 669 is id for whole visual cortex
    indices_VIS = np.argwhere(VIS_mask == 1)
    # select anterior and posterior parts of A1
    max_y_vis = np.max(indices_VIS[:, 0])
    min_y = np.min(indices_VIS[:, 0])
    AP_midpoint_VIS = ((max_y_vis - min_y) / 2) + min_y
    x_midpoint = VIS_mask.shape[2] // 2
    contra_mask = np.zeros_like(VIS_mask, dtype=bool)
    contra_mask[:, :, x_midpoint:] = 1
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    AUDp_id = bg_atlas.structures["AUDp"]["id"]

    rsp = mcc.get_reference_space()
    AUDp_mask = rsp.make_structure_mask([AUDp_id], direct_only=False)
    indices_AUDp = np.argwhere(AUDp_mask == 1)

    # select anterior and posterior parts of A1
    max_y = np.max(indices_AUDp[:, 0])
    min_y = np.min(indices_AUDp[:, 0])
    AP_midpoint_A1 = ((max_y - min_y) / 2) + min_y
    # now select only the ipsiliateral side of where was injected
    x_midpoint = AUDp_mask.shape[2] // 2
    contra_mask = np.zeros_like(AUDp_mask, dtype=bool)
    contra_mask[:, :, x_midpoint:] = 1

    # now lets load the barcodes
    proj_path = Path(gen_parameters["proj_path"])
    mice = gen_parameters["MICE"]
    mouse_dict_AP_source = {}
    # mouse_barcodes_by_source = {}
    mouse_dict_A1_coords = {}
    for mouse in mice:
        AP_position_dict = {}
        A1_coord_dict = {}
        barcodes = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded_with_source.pkl"
        )
        lcm_directory = proj_path / f"{mouse}/LCM"
        ROI_3D = np.load(lcm_directory / "ROI_3D_25.npy")
        # AP_samples = {}
        # AP_source_filtered = {}
        all_AUDp_samples = np.unique(ROI_3D * AUDp_mask * contra_mask)
        all_AUDp_samples = [sample for sample in all_AUDp_samples if sample != 0]
        all_AUDp_samples = [
            sample for sample in all_AUDp_samples if sample in barcodes.columns
        ]
        for sample in all_AUDp_samples:
            centroid = np.argwhere(ROI_3D == sample).mean(axis=0)
            AP_position_dict[sample] = max_y_vis - centroid[0]
            A1_coord_dict[sample] = centroid
        mouse_dict_AP_source[mouse] = AP_position_dict
        mouse_dict_A1_coords[mouse] = A1_coord_dict
    return mouse_dict_AP_source, mouse_dict_A1_coords


def find_soma_AP_roi(
    mice, proj_path, mouse_dict_AP_source, mouse_dict_A1_coords, roi_dict
):
    # we only want to look at cortical projecting neurons, therefore, we filter by the cortical areas covered in our dataset
    cortical_areas = [
        "VISp",
        "VISpl",
        "VISli",
        "ACAv",
        "TEa",
        "SSs",
        "AUDpo",
        "PERI",
        "VISal",
        "RSPv",
        "VISl",
        "MOp",
        "VISpm",
        "AUDp",
        "VISa",
        "RSPd",
        "VISpor",
        "SSp",
        "AUDd",
        "AUDv",
        "MOs",
        "RSPagl",
        "ACAd",
        "VISrl",
        "ECT",
        "VISam",
        "Contra",
    ]
    all_AP_dict = {}
    # vis_mean_dict = {}
    # barcodes_per_sample = {}
    which_soma = pd.DataFrame(columns=["Mouse", "Sample", "AP_position", "Coords"])
    cortical_roi_dict = {}
    for mouse in mice:
        # sample_num_dict = {}
        sample_3d = roi_dict[mouse].copy()
        barcodes = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded_with_source.pkl"
        )
        lcm_dir = f"{proj_path}/{mouse}/LCM"
        area_matrix = get_area_volumes(
            barcode_table_cols=barcodes.columns,
            lcm_directory=pathlib.Path(lcm_dir),
            area_threshold=0.1,
        )
        cortical_samples = area_matrix[
            area_matrix[cortical_areas].astype(bool).sum(axis=1) > 0
        ].index.to_list()
        frac = area_matrix.div(area_matrix.sum(axis=1), axis=0)
        frac_filtered = frac.loc[
            (frac[cortical_areas].gt(0).any(axis=1)) & (frac["AUDp"] > 0.1)
        ].index
        all_VIS_ROI = [
            sample
            for sample in cortical_samples
            if sample in barcodes.columns and sample not in frac_filtered
        ]
        mask = np.isin(sample_3d, all_VIS_ROI)
        sample_3d[~mask] = 0
        cortical_roi_dict[mouse] = sample_3d
        sample_dict = {}
        the_soma_samples = []
        for sample in cortical_samples:
            bcs = barcodes.copy()
            vis_proj = bcs[bcs[sample].astype(bool) > 0]
            soma = pd.DataFrame(vis_proj.idxmax(axis=1))
            soma["AP_position"] = soma.apply(
                lambda row: get_AP_position(row, mouse_dict_AP_source[mouse]), axis=1
            )
            sample_dict[sample] = soma["AP_position"]
            for soma_sample in vis_proj.idxmax(axis=1).unique():
                if soma_sample not in the_soma_samples:
                    the_soma_samples.append(soma_sample)
        for sample_to_look in the_soma_samples:
            new_row = pd.DataFrame(
                {
                    "Mouse": [mouse],
                    "Sample": [sample_to_look],
                    "AP_position": [mouse_dict_AP_source[mouse][sample_to_look] * 25],
                    "Coords": [mouse_dict_A1_coords[mouse][sample_to_look]],
                }
            )
            which_soma = pd.concat([which_soma, new_row])
        all_AP_dict[mouse] = sample_dict

    which_soma.reset_index(inplace=True)
    return cortical_roi_dict, all_AP_dict, which_soma


def build_AP_soma_3d(
    cortical_roi_dict, roi_dict, all_AP_dict, mice, which_soma, proj_top
):
    arrays = list(cortical_roi_dict.values())
    bool_arrays = [(arr > 0) for arr in arrays]
    combined_bool_any = np.any(bool_arrays, axis=0)
    coords = np.argwhere(combined_bool_any)
    some_mouse = mice[0]
    master_3d = np.full(cortical_roi_dict[some_mouse].shape, np.nan, dtype=float)
    for i, j, k in coords:  # for each pixel compute mean soma AP position
        value_list = []
        for mouse in mice:
            roi_to_look = cortical_roi_dict[mouse]
            sample = roi_to_look[i, j, k]
            if sample > 0:
                list_of_pos = all_AP_dict[mouse][sample].to_list()
                if len(list_of_pos) > 2:
                    value_list.extend(list_of_pos)
        master_3d[i, j, k] = np.mean(value_list) * 25

    which_soma["2d_coords_y"] = None
    which_soma["2d_coords_x"] = None
    for mouse in mice:
        mouse_2d = proj_top.project_volume(roi_dict[mouse])
        to_looksoma = mouse_2d.T
        to_looksoma = np.flip(to_looksoma, axis=1)
        for ind, row in which_soma.iterrows():
            if row["Mouse"] == mouse:
                sample = row["Sample"]
                where_the_sample = np.argwhere(to_looksoma == sample)
                which_2d_coord = np.mean(where_the_sample, axis=0)
                which_soma.loc[ind, "2d_coords_y"] = which_2d_coord[0]
                which_soma.loc[ind, "2d_coords_x"] = which_2d_coord[1]

    return master_3d, which_soma


def plot_AP_soma_cubelet_flatmap(
    gen_parameters,
    fig,
    ax,
    roi_dict,
    proj_top,
    bf_left_boundaries,
    bf_right_boundaries,
):
    mouse_dict_AP_source, mouse_dict_A1_coords = get_AP_coords(
        gen_parameters=gen_parameters
    )
    cortical_roi_dict, all_AP_dict, which_soma = find_soma_AP_roi(
        mice=gen_parameters["MICE"],
        proj_path=gen_parameters["proj_path"],
        mouse_dict_AP_source=mouse_dict_AP_source,
        mouse_dict_A1_coords=mouse_dict_A1_coords,
        roi_dict=roi_dict,
    )
    master_3d, which_soma = build_AP_soma_3d(
        roi_dict=roi_dict,
        cortical_roi_dict=cortical_roi_dict,
        all_AP_dict=all_AP_dict,
        mice=gen_parameters["MICE"],
        which_soma=which_soma,
        proj_top=proj_top,
    )
    clean_volume = np.nan_to_num(master_3d, nan=-np.inf)
    clean_volume_max = proj_top.project_volume(clean_volume)
    to_look = clean_volume_max.T
    to_look = np.flip(to_look, axis=1)
    cmap = plt.cm.get_cmap("rainbow").copy()
    cmap.set_bad(color=[0.3, 0.3, 0.3, 1])  #
    to_look_masked = np.where(to_look <= 0, np.nan, to_look)
    im = ax.imshow(to_look_masked, cmap=cmap, vmin=1500, vmax=clean_volume_max.max())
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Mean soma AP position (µm)", fontsize=6, family="Arial")
    cbar.ax.tick_params(labelsize=6, labelrotation=0)
    for l, boundary_coords in bf_left_boundaries.items():
        ax.plot(*boundary_coords.T, c="white", lw=0.3)
    for l, boundary_coords in bf_right_boundaries.items():
        ax.plot(*boundary_coords.T, c="white", lw=0.3)
    coords_y = which_soma["2d_coords_y"].to_numpy()
    coords_x = which_soma["2d_coords_x"].to_numpy()
    colors = which_soma["AP_position"].to_numpy()
    ax.scatter(
        coords_x,
        coords_y,
        c=colors,
        cmap=cmap,
        vmin=colors.min(),
        vmax=clean_volume_max.max(),
        s=1.5,
        edgecolors="black",
        linewidths=0.15,
        label="Soma centroids",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def homog_across_cubelet(
    parameters_path,
    cortical,
    shuffled,
    barcode_matrix,
    CT_PT_only=False,
    IT_only=False,
    area_threshold=0.1,
    binary=False,
):
    """
    Function to output a matrix of homogenous across areas, looking only at cortical samples
    Args:
        parameters_path
        barcode_matrix = pandas dataframe with barcodes
        cortical (bool): True if you want onkly to look at cortical regions
        shuffled (bool): True if you want to shuffle values in all columns as a negative control
        CT_PT_only (bool): True if you just want to look at corticothalamic/pyramidal tract neurons
        IT_only (bool): True if you want to look at only intratelencephalic neurons
    """
    parameters = lf.load_parameters(directory=parameters_path)
    # sequencing_directory = pathlib.Path(
    #     "".join(
    #         [
    #             parameters["PROCESSED_DIR"],
    #             "/",
    #             parameters["PROJECT"],
    #             "/",
    #             parameters["MOUSE"],
    #             "/Sequencing",
    #         ]
    #     )
    # )
    barcodes_across_sample = barcode_matrix.copy()
    proj_path = parameters_path.split("/Sequencing")[0]
    lcm_directory = Path(f"{proj_path}/LCM")
    # lcm_directory = parameters["lcm_directory"]
    cortical_samples_columns = [
        int(col)
        for col in parameters["cortical_samples"]
        if col in barcodes_across_sample.columns
    ]
    # only look at cortical samples
    if cortical:
        barcodes_across_sample = barcodes_across_sample[cortical_samples_columns]
    if CT_PT_only or IT_only:
        sample_path = lcm_directory / "sample_vol_and_regions.pkl"
        sample_vol_and_regions = pd.read_pickle(sample_path)
        sample_vol_and_regions["regions"] = sample_vol_and_regions["regions"].apply(
            ast.literal_eval
        )
        roi_list = []
        index_list = []
        for index, row in sample_vol_and_regions.iterrows():
            if any(
                "IC" in region
                or "SCs" in region
                or "SCm" in region
                or "LGd" in region
                or "LGv" in region
                or "MGv" in region
                or "RT" in region
                or "LP" in region
                or "MGd" in region
                or "P," in region
                for region in row["regions"]
            ):
                if row["ROI Number"] not in parameters["cortical_samples"]:
                    index_list.append(index)
                    roi_list.append(row["ROI Number"])
        roi_list = [
            sample for sample in roi_list if sample in barcodes_across_sample.columns
        ]
        if CT_PT_only:
            barcodes_across_sample = barcodes_across_sample[
                barcodes_across_sample[roi_list].sum(axis=1) > 0
            ]
        if IT_only:
            barcodes_across_sample = barcodes_across_sample[
                barcodes_across_sample[roi_list].sum(axis=1) == 0
            ]
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample.astype(bool).sum(axis=1) > 0
    ]

    areas_only_grouped = get_area_volumes(
        barcode_table_cols=barcodes_across_sample.columns,
        lcm_directory=pathlib.Path(lcm_directory),
        area_threshold=area_threshold,
    )
    zero_cols = areas_only_grouped.index[
        (areas_only_grouped == 0).all(axis=1, skipna=True)
    ].tolist()
    areas_only_grouped.drop(zero_cols, inplace=True)
    barcodes_across_sample.drop(columns=zero_cols, inplace=True)
    areas_matrix = areas_only_grouped.to_numpy()
    total_frac = np.sum(areas_matrix, axis=1)
    frac_matrix = areas_matrix / total_frac[:, np.newaxis]
    weighted_frac_matrix = frac_matrix / areas_matrix.sum(axis=0)
    barcodes = barcodes_across_sample.to_numpy()
    if shuffled and not binary:
        barcodes = send_to_shuffle(barcodes=barcodes)
    if binary and shuffled:
        barcodes = send_for_curveball_shuff(
            barcodes=barcodes
        )  # note this is not used for flatmaps
    bc_matrix = np.matmul(barcodes, weighted_frac_matrix)
    bc_matrix = pd.DataFrame(
        data=bc_matrix,
        columns=areas_only_grouped.columns.to_list(),
        index=barcodes_across_sample.index,
    )

    bc_matrix = bc_matrix.dropna(axis=1, how="all")
    bc_matrix = bc_matrix.loc[~(bc_matrix == 0).all(axis=1)]
    row_min = bc_matrix.min(axis=1)
    row_range = bc_matrix.max(axis=1) - row_min
    row_range.replace(0, np.nan, inplace=True)
    bc_matrix = bc_matrix.sub(row_min, axis=0)
    bc_matrix = bc_matrix.div(
        row_range, axis=0
    )  # subtract min and divide by range so max = 1
    if binary:
        bc_matrix = bc_matrix.astype(bool).astype(int)
    return bc_matrix.fillna(0)


def normalize_barcodes(barcodes):
    # since we have the soma barcode here, we normalise to 1 for the second max (non-som, max projection) value, then set the soma to 1
    # edit: we now make it so max is 1
    normalized_barcodes = barcodes.copy().astype(float)
    for index, row in normalized_barcodes.iterrows():
        row_values = row.values
        min_val = row_values.min()
        sorted_vals = sorted(row_values)
        unique_vals = sorted(set(row_values))
        if len(unique_vals) < 2:
            # if only one unique value (i.e. the soma) — set row to NaN - shouldn't be the case
            normalized_barcodes.loc[index] = np.nan
            continue
        second_max = unique_vals[-2]
        denominator = second_max - min_val
        assert min_val == 0, f"Expected min to be 0, got {min_val}"
        assert second_max > 0, f"Expected second max > 0, got {second_max}"
        normalized_row = (row_values - min_val) / denominator
        normalized_barcodes.loc[index] = normalized_row
    return normalized_barcodes


def plot_indiv_projections(
    sample_vol_and_regions,
    roi_dict,
    barcodes_dict,
    gen_parameters,
    fig,
    proj_top,
    bf_left_boundaries,
    bf_right_boundaries,
):
    proj_path = gen_parameters["proj_path"]
    combined_barcodes_mouse, vis_barcodes = compile_vis_barcodes_dfs(
        sample_vol_and_regions=sample_vol_and_regions,
        roi_dict=roi_dict,
        barcodes_dict=barcodes_dict,
        gen_parameters=gen_parameters,
    )
    mice = gen_parameters["MICE"]
    font_size = gen_parameters["font_size"]
    convert_dict = get_convert_dict()
    order_in_other_plot = [
        "AUDd",
        "AUDv",
        "AUDpo",
        "TEa",
        "PERI",
        "ECT",
        "VISC",
        "VISal",
        "VISp",
        "VISpl",
        "VISpm",
        "VISrl",
        "VISpor",
        "VISam",
        "VISl",
        "VISa",
        "VISli",
        "MOp",
        "MOs",
        "SSp",
        "SSs",
        "RSPv",
        "RSPd",
        "RSPagl",
        "ACAd",
        "ACAv",
        "Contra",
    ]
    order_in_other_plot = [
        convert_dict[val] if val in convert_dict else val for val in order_in_other_plot
    ]
    gen_ind = 1964
    all_ind = 309
    dorsal_ind = 205
    dict_of_values = {}
    gs = GridSpec(
        nrows=2, ncols=3, figure=fig, height_ratios=[3, 1], hspace=0.12, wspace=0.3
    )

    for col, bc_index in enumerate([dorsal_ind, all_ind, gen_ind]):
        roi_and_barchart = {}
        mouse = combined_barcodes_mouse.iloc[bc_index]["mouse"]
        lengths = [len(vis_barcodes[m]) for m in mice]
        cumulative_lengths = [0] + list(np.cumsum(lengths))
        mouse_idx = mice.index(mouse)
        adjusted_bc_ind = bc_index - cumulative_lengths[mouse_idx]
        barcodes_across_sample = vis_barcodes[mouse]
        roi_to_look = roi_dict[mouse]
        vals = np.unique(roi_to_look)
        values_not_in_barcodes = vals[~np.isin(vals, barcodes_across_sample.columns)]
        mask = ~np.isin(roi_to_look, values_not_in_barcodes)
        roi_to_look = np.where(mask, roi_to_look, 0)

        all_flatmap = proj_top.project_volume(roi_to_look)
        rebuilding_all_flatmap = np.where(all_flatmap == 0, -1.0, 0.0)

        bc = barcodes_across_sample.copy()
        vol_series = sample_vol_and_regions[mouse].set_index("ROI Number")[
            "Volume (um^3)"
        ]
        volumnes_df = vol_series.to_frame().T
        volumnes_df = volumnes_df[bc.columns]
        density_df = bc.div(volumnes_df.iloc[0], axis=1)
        normalised_density_df = normalize_barcodes(density_df)
        val_list = []
        all_rois = np.unique(roi_to_look)
        for roi in bc.columns:
            density = normalised_density_df.iloc[adjusted_bc_ind][roi]
            if density > 0 and (roi in all_rois):
                val_list.append(density)
                binary_roi = np.where(roi_to_look == roi, density, 0.0)
                flat_projection = proj_top.project_volume(binary_roi)
                rebuilding_all_flatmap[rebuilding_all_flatmap == 0] = flat_projection[
                    rebuilding_all_flatmap == 0
                ]

        smallest_value = 1e-3
        # bl = rebuilding_all_flatmap.copy()
        no_val_vals = rebuilding_all_flatmap == -1
        rebuilding_all_flatmap[no_val_vals] = -smallest_value
        rebuilding_all_flatmap = rebuilding_all_flatmap + smallest_value
        density_map = np.where(
            rebuilding_all_flatmap == -1, np.nan, rebuilding_all_flatmap
        )
        data = np.flip(density_map.T, axis=1)
        ax_map = fig.add_subplot(gs[0, col])
        finite_vals = data[np.isfinite(data)]
        max_value = np.max(finite_vals)
        second_max_value = (
            np.max(finite_vals[finite_vals < max_value])
            if np.any(finite_vals < max_value)
            else max_value
        )
        vmin = smallest_value
        vmax = second_max_value
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        roi_and_barchart["roi"] = data

        magma = plt.cm.magma
        plot_cmap = magma.copy()
        plot_cmap.set_over("chartreuse")
        plot_cmap.set_bad("white")
        im = ax_map.imshow(data, cmap=plot_cmap, norm=norm, interpolation="nearest")
        ax_map.set_title(f"Neuron {bc_index}", fontsize=font_size * 1.1, pad=4)
        ax_map.axis("off")
        for coords in bf_left_boundaries.values():
            ax_map.plot(*coords.T, c="grey", lw=0.3)
        for coords in bf_right_boundaries.values():
            ax_map.plot(*coords.T, c="grey", lw=0.3)

        cbar = fig.colorbar(
            im,
            ax=ax_map,
            fraction=0.046,
            pad=0.02,
        )

        cbar.set_label(
            "Normalised projection density", fontsize=font_size
        )  # Previously "Normalised Counts/mm³"
        cbar.ax.tick_params(labelsize=font_size)
        cbar.locator = LogLocator(base=10)
        cbar.update_ticks()
        ax_bar = fig.add_subplot(gs[1, col])
        parameters_path = f"{proj_path}/{mouse}/Sequencing/"
        AUD_source = sample_vol_and_regions[mouse][
            sample_vol_and_regions[mouse].main.str.contains("AUDp")
        ]["ROI Number"].to_list()
        AUD_source = [s for s in AUD_source if s in barcodes_across_sample.columns]
        barcodes_no_aud = barcodes_across_sample.drop(columns=AUD_source)

        bc_area = homog_across_cubelet(
            parameters_path=parameters_path,
            barcode_matrix=barcodes_no_aud,
            cortical=True,
            shuffled=False,
            binary=False,
            IT_only=True,
            area_threshold=0.1,
        )

        bc_area = convert_matrix_names(bc_area)

        labels = bc_area.iloc[adjusted_bc_ind].index.astype(str).tolist()
        values = bc_area.iloc[adjusted_bc_ind].values
        # added in scaling so it's between 0 and 1
        ordered_labels = [val for val in order_in_other_plot if val in labels]
        label_value_dict_scaled = bc_area.iloc[adjusted_bc_ind][
            ordered_labels
        ].to_dict()
        label_value_dict = dict(zip(labels, values))

        ordered_values = [label_value_dict_scaled[label] for label in ordered_labels]
        what_plot = ax_bar.bar(ordered_labels, ordered_values, color="mediumpurple")
        roi_and_barchart["barchart"] = label_value_dict_scaled
        dict_of_values[mouse] = roi_and_barchart

        ff.myPlotSettings_splitAxis(
            fig=what_plot, ax=ax_bar, ytitle="", xtitle="", title="", mySize=font_size
        )
        if col == 0:
            ax_bar.set_ylabel(
                "Normalised area\nprojection density", fontsize=font_size, labelpad=4
            )
        else:
            ax_bar.set_ylabel("")
        ax_bar.tick_params(axis="x", labelsize=font_size * 0.9, labelrotation=90)
        ax_bar.tick_params(axis="y", labelsize=font_size)
        ax_bar.set_xlim(-0.5, len(ordered_labels) - 0.5)
        ax_bar.set_yticks([0.0, 0.5, 1])
        ax_bar.set_yticklabels(["0.0", "0.5", "1.0"])
    chartreuse_patch = Patch(facecolor="chartreuse", label="soma cubelet")
    fig.legend(
        handles=[chartreuse_patch],
        loc="upper left",
        fontsize=font_size,
        bbox_to_anchor=(0.07, 0.9),
        frameon=False,
    )

    plt.show()


def get_convert_dict():
    convert_dict = {
        "VISl": "LM",
        "VISrl": "RL",
        "VISal": "AL",
        "VISa": "A",
        "VISp": "V1",
        "VISpor": "POR",
        "VISli": "LI",
        "VISpl": "P",
        "VISpm": "PM",
        "VISam": "AM",
    }
    return convert_dict


def convert_matrix_names(matrix):
    convert_dict = {
        "VISp": "V1",
        "VISpor": "POR",
        "VISli": "LI",
        "VISal": "AL",
        "VISl": "LM",
        "VISpl": "P",
        "VISpm": "PM",
        "VISrl": "RL",
        "VISam": "AM",
        "VISa": "A",
    }
    matrix.rename(columns=convert_dict, inplace=True)
    matrix.rename(index=convert_dict, inplace=True)
    return matrix


def compile_vis_barcodes_dfs(
    sample_vol_and_regions, roi_dict, barcodes_dict, gen_parameters
):
    mice = gen_parameters["MICE"]
    combined_samples = None
    for mouse in mice:
        if combined_samples is None:
            combined_samples = roi_dict[mouse] > 0
        else:
            combined_samples |= roi_dict[mouse] > 0
    combined_samples = combined_samples.astype(int)
    vis_barcodes = {}
    for mouse in mice:
        VIS_samples = sample_vol_and_regions[mouse][
            sample_vol_and_regions[mouse].main.str.contains("VIS")
        ]["ROI Number"].to_list()
        VIS_samples = [
            sample for sample in VIS_samples if sample in barcodes_dict[mouse].columns
        ]
        barcodes_across_sample = barcodes_dict[mouse]
        vis_barcodes[mouse] = barcodes_dict[mouse][
            barcodes_dict[mouse][VIS_samples].ne(0).any(1)
        ]

    combined_mouse_labels = []
    for mouse in mice:
        count = len(vis_barcodes[mouse])
        combined_mouse_labels.extend([mouse] * count)
    combined_barcodes_mouse = pd.DataFrame({"mouse": combined_mouse_labels})
    return combined_barcodes_mouse, vis_barcodes
