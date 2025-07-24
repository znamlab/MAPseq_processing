# from preprocessing_sequencing import preprocess_sequences as ps
from znamutils import slurm_it
import pandas as pd
import numpy as np
from pathlib import Path
import ast
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import itertools
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import os
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from random import sample, shuffle
from concurrent.futures import ProcessPoolExecutor
import subprocess
from final_processing import loading_functions as lf
from flexiznam.config import PARAMETERS
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp, pearsonr
from dataclasses import dataclass
from typing import Tuple
from scipy.stats import norm
from bg_atlasapi import BrainGlobeAtlas
import statsmodels.formula.api as smf
import final_processing.helper_functions as hf
from scipy.stats import norm
from statsmodels.stats.contrast import ContrastResults


def samples_to_areas(mice, proj_path, shuffled=False, binary=False):
    """function to generate a dictionary of with mice as keys for neuron barcodes across areas (from neuron barcodes across samples)"""
    combined_dict = {}
    for num, mouse in enumerate(mice):
        new_dict = {}
        parameters_path = f"{proj_path}/{mouse}/Sequencing"
        barcodes = pd.read_pickle(f"{parameters_path}/A1_barcodes_thresholded.pkl")
        barcodes = add_prefix_to_index(barcodes, mouse)
        new_dict["homogenous_across_cubelet"] = homog_across_cubelet(
            parameters_path=parameters_path,
            barcode_matrix=barcodes,
            cortical=True,
            IT_only=True,
            shuffled=shuffled,
            binary=binary,
        )
        new_dict["max_counts"] = barcodes.max(axis=1)
        combined_dict[mouse] = new_dict
    return combined_dict


def homog_across_cubelet(
    parameters_path,
    cortical,
    shuffled,
    barcode_matrix,
    CT_PT_only=False,
    IT_only=False,
    area_threshold=0.1,
    binary=False,
    remove_AUDp_vis_cub=False,
    run_externally=True,
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

    # barcodes_across_sample = pd.read_pickle(sequencing_directory / "A1_barcodes.pkl")
    barcodes_across_sample = barcode_matrix.copy()
    if run_externally == False:
        processed_path = Path(PARAMETERS["data_root"]["processed"])
        lcm_directory = processed_path / (
            "turnerb_" + parameters["lcm_directory"].split("turnerb_")[1]
        )
    else:
        proj_path = parameters_path.split("/Sequencing")[0]
        lcm_directory = Path(f"{proj_path}/LCM")
    cortical_samples_columns = [
        int(col)
        for col in parameters["cortical_samples"]
        if col in barcodes_across_sample.columns
    ]
    # only look at cortical samples
    if cortical:
        barcodes_across_sample = barcodes_across_sample[cortical_samples_columns]
    if CT_PT_only or IT_only:
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
        lcm_directory=lcm_directory,
        area_threshold=area_threshold,
    )
    if remove_AUDp_vis_cub == True:
        visual_areas = [
            "VISli",
            "VISpor",
            "VISpl",
            "VISl",
            "VISp",
            "VISal",
            "VISam",
            "VISpm",
            "VISa",
            "VISrl",
        ]
        frac = areas_only_grouped.div(areas_only_grouped.sum(axis=1), axis=0)
        frac_filtered = frac.loc[
            (frac[visual_areas].gt(0).any(axis=1)) & (frac["AUDp"] > 0.1)
        ].index
        barcodes_across_sample.drop(columns=frac_filtered, inplace=True)
        areas_only_grouped = areas_only_grouped.drop(
            index=frac_filtered, errors="ignore"
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
        barcodes = send_for_curveball_shuff(barcodes=barcodes)
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

    # bc_matrix = bc_matrix.sub(row_min, axis=0)
    bc_matrix = bc_matrix.div(row_range, axis=0)

    if binary:
        bc_matrix = bc_matrix.astype(bool).astype(int)
    return bc_matrix.fillna(0)


def send_for_curveball_shuff(barcodes):
    """Function to perform shuffling using curvball algorithm (shuffling where binarized column and row sums stay the same)
    Args:
        barcodes: pd.dataframe containing barcodes across cubelets
    Returns:
        shuffled matrix
    """
    barcodes = barcodes.astype(bool).astype(int)
    presences = find_presences(barcodes)
    r_presences = presences[:]
    iter = 5 * len(barcodes)  # number of iterations of pairs of rows shuffling
    return curve_ball(barcodes, r_presences, num_iterations=iter)


def find_presences(input_matrix):
    num_rows, num_cols = input_matrix.shape
    hp = []
    iters = num_rows if num_cols >= num_rows else num_cols
    input_matrix_b = (
        input_matrix if num_cols >= num_rows else np.transpose(input_matrix)
    )
    for r in range(iters):
        hp.append(list(np.where(input_matrix_b[r] == 1)[0]))
    return hp


def curve_ball(input_matrix, r_hp, num_iterations=-1):
    num_rows, num_cols = input_matrix.shape
    l = range(len(r_hp))
    num_iters = 5 * min(num_rows, num_cols) if num_iterations == -1 else num_iterations
    for rep in range(num_iters):
        AB = sample(l, 2)
        a = AB[0]
        b = AB[1]
        ab = set(r_hp[a]) & set(r_hp[b])  # common elements
        l_ab = len(ab)
        l_a = len(r_hp[a])
        l_b = len(r_hp[b])
        if l_ab not in [l_a, l_b]:
            tot = list(set(r_hp[a] + r_hp[b]) - ab)
            ab = list(ab)
            shuffle(tot)
            L = l_a - l_ab
            r_hp[a] = ab + tot[:L]
            r_hp[b] = ab + tot[L:]
    out_mat = (
        np.zeros(input_matrix.shape, dtype="int8")
        if num_cols >= num_rows
        else np.zeros(input_matrix.T.shape, dtype="int8")
    )
    for r in range(min(num_rows, num_cols)):
        out_mat[r, r_hp[r]] = 1
    result = out_mat if num_cols >= num_rows else out_mat.T
    return result


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
        # all_regions = sample_vol_and_regions_FIAA456d.loc[sample_vol_and_regions_FIAA456d.index[sample_vol_and_regions_FIAA456d['ROI Number'] == column].tolist(), 'Brain Regions'].explode().astype(int)
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


def send_to_shuffle(barcodes):
    """
    Function to randomly shuffle values in all columns of barcode matrix as a negative control
    Args:
        barcodes: numpy matrix of neuron barcodes
    Returns:
        column shuffled matrix
    """
    shuffled_counts = np.copy(barcodes)
    for i in range(shuffled_counts.shape[1]):
        np.random.shuffle(shuffled_counts[:, i])
    return shuffled_counts


def add_prefix_to_index(df, prefix):
    df = df.copy()
    df.index = [f"{prefix}_{idx}" for idx in df.index]
    return df


def get_common_columns(mice, combined_dict, cortex, key="homogenous_across_cubelet"):
    """Function to get common areas across mouse barcode dictionaries. If cortex ==True, only take cortical areas"""
    mcc = MouseConnectivityCache(resolution=25)
    structure_tree = mcc.get_structure_tree()
    common_columns = set.intersection(
        *[set(combined_dict[k][key].columns) for k in mice]
    )
    # let's make sure that all the areas are cortical (areas such as HPF are unintentially side bits of cubelets and never main target, and more likely registration errors)
    if cortex:
        common_cols_cortex = []
        for col in common_columns:
            if col == "Contra":
                common_cols_cortex.append(col)
            if col not in ["Contra", "OB"]:
                structure = structure_tree.get_structures_by_acronym([col])
                if 315 in structure[0]["structure_id_path"]:
                    common_cols_cortex.append(col)
    return common_cols_cortex if cortex else common_columns


def exponential(x, a, b):
    return a * np.exp(-b * x)


@dataclass
class FitResult:
    params: Tuple[float, float]
    ks_stat: float
    ks_p: float
    cv_corr: float
    cv_p: float
    fitted_x: np.ndarray
    fitted_y: np.ndarray


def get_means_errors(df: pd.DataFrame):
    return pd.to_numeric(df.mean(), errors="coerce"), pd.to_numeric(
        df.std(), errors="coerce"
    )


def fit_exponential(distances: pd.Series, means: pd.Series, p0=(1, 0.001)) -> FitResult:
    params, _ = curve_fit(exponential, distances, means, p0=p0)
    fitted_x = np.linspace(distances.min(), distances.max(), 100)
    fitted_y = exponential(fitted_x, *params)

    # goodness-of-fit metrics
    fitted_vals = exponential(distances, *params)
    ks_stat, ks_p = ks_2samp(means, fitted_vals)

    # leave-one-out cross-validation
    loo_preds = means.copy()
    for i in range(len(distances)):
        mask = np.ones(len(distances), dtype=bool)
        mask[i] = False
        params_i, _ = curve_fit(exponential, distances[mask], means[mask], p0=p0)
        loo_preds.iloc[i] = exponential(distances.iloc[i], *params_i)
    cv_corr, cv_p = pearsonr(means, loo_preds)

    return FitResult(params, ks_stat, ks_p, cv_corr, cv_p, fitted_x, fitted_y)


def get_distances_from_A1(combined_dict, area_cols, mice):
    """function to make a dataframe of distances from A1"""
    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()
    rsp = mcc.get_reference_space()
    # a1_dist_dict = {}
    structure = structure_tree.get_structures_by_acronym(["AUDp"])
    structure_id = structure[0]["id"]
    mask = rsp.make_structure_mask([structure_id], direct_only=False)
    A1_coord = (
        np.mean(np.where(mask == 1)[0]),
        np.mean(np.where(mask == 1)[1]),
        np.mean(np.where(mask == 1)[2]),
    )
    key_to_plot = "homogenous_across_cubelet"
    areas = area_cols.drop(["Contra", "AUDp"])
    # vis_adj = [vis for vis in visual_areas if vis in all_mice[key_to_plot].columns]
    distance_from_a1 = pd.DataFrame(index=areas, columns=["dist"])
    for col in areas:
        structure = structure_tree.get_structures_by_acronym([col])
        structure_id = structure[0]["id"]
        mask = rsp.make_structure_mask([structure_id], direct_only=False)
        vis_coord = (
            np.mean(np.where(mask == 1)[0]),
            np.mean(np.where(mask == 1)[1]),
            np.mean(np.where(mask == 1)[2]),
        )
        distance_from_a1.loc[col] = (
            np.linalg.norm(np.array(A1_coord) - np.array(vis_coord)) * 25
        )

    # a1_dist_dict[key_to_plot] = distance_from_a1
    freq_df = pd.DataFrame(columns=areas, index=mice)
    freq_df_strength = pd.DataFrame(columns=areas, index=mice)
    for mouse in mice:
        freq_df.loc[mouse] = combined_dict[mouse][key_to_plot][areas].astype(bool).sum(
            axis=0
        ) / len(combined_dict[mouse][key_to_plot])
        freq_df_strength.loc[mouse] = (
            combined_dict[mouse][key_to_plot][areas]
            .where(combined_dict[mouse][key_to_plot][areas] > 0)
            .mean(axis=0)
        )
    distances = pd.Series(distance_from_a1.iloc[:, 0], index=areas)
    distances = pd.to_numeric(distances, errors="coerce")
    return freq_df, freq_df_strength, distances


def get_contra_mask(mask_shape):
    contra = np.zeros(mask_shape, dtype=bool)
    contra[:, :, mask_shape[2] // 2 :] = 1
    return contra


def get_area_mask(area_id):
    """Function to get mask in allen ccf of area ids
    Args:
    area_id: (int) numerical id of area
    """
    mcc = MouseConnectivityCache()
    rsp = mcc.get_reference_space()
    area_mask = rsp.make_structure_mask([area_id], direct_only=False)
    return area_mask


def get_AUDp_VIS_coords():
    """Function to use allen brain ccf and extract min, max, centroids of the primary auditory cortex, and whole visual cortex"""
    # mcc = MouseConnectivityCache()
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    # AUDp_id = bg_atlas.structures['AUDp']['id']
    # VIS_id = bg_atlas.structures['VIS']['id']
    # rsp = mcc.get_reference_space()
    AUDp_mask = get_area_mask(
        area_id=bg_atlas.structures["AUDp"]["id"]
    )  # rsp.make_structure_mask([AUDp_id], direct_only=False)
    indices_AUDp = np.argwhere(AUDp_mask == 1)
    VIS_mask = get_area_mask(
        bg_atlas.structures["VIS"]["id"]
    )  # area_id=669) #rsp.make_structure_mask([669], direct_only=False) #669 is id for whole visual cortex
    indices_VIS = np.argwhere(VIS_mask == 1)
    max_y_vis = np.max(indices_VIS[:, 0])
    min_y_vis = np.min(indices_VIS[:, 0])
    # select anterior and posterior parts of A1
    max_y = np.max(indices_AUDp[:, 0])
    min_y = np.min(indices_AUDp[:, 0])
    # AP_midpoint_A1 = ((max_y - min_y) /2) + min_y
    # posterior_neurons = indices_AUDp[indices_AUDp[:, 0]>=AP_midpoint_A1]
    # anterior_neurons = indices_AUDp[indices_AUDp[:, 0]<AP_midpoint_A1]
    # now select only the ipsiliateral side of where was injected
    # x_midpoint = AUDp_mask.shape[2] // 2
    contra_mask = get_contra_mask(AUDp_mask.shape)
    # contra_mask = np.zeros_like(AUDp_mask, dtype=bool)
    # contra_mask[:, :, x_midpoint:] = 1
    # lets get the coordinates for the centre of A1
    A1_masked = contra_mask * AUDp_mask
    A1_centroid_coords = np.argwhere(A1_masked == 1).mean(axis=0)
    return max_y, min_y, max_y_vis, min_y_vis, A1_centroid_coords


def get_AP_positioning_cubelets(mice, proj_path, HVA_cols, max_y_vis):
    """Function to create dictionaries for A-P coordinate positions of A1 soma cubelets and visual cortex target cubelets
    Args:
        mice: list of mice ids used for finding paths
        proj_path: path to where pre-processed MAPseq datasets are stored
        HVA_cols: list of visual cortex areas we use to limit analysis to cubelets in these areas
        max_y_vis: most posterior coordinate in visual cortex (used to normalise A-P values)
    """
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    proj_path = Path(proj_path)
    mouse_dict_AP_source = {}
    mouse_dict_AP_VC = {}
    mouse_vis_main_dict = {}
    mouse_vis_coord = {}
    AP_positioning_dicts = {}
    mouse_dict_AP_source_coords = {}

    AUDp_mask = get_area_mask(area_id=bg_atlas.structures["AUDp"]["id"])
    contra_mask = get_contra_mask(AUDp_mask.shape)
    VIS_mask = get_area_mask(bg_atlas.structures["VIS"]["id"])
    for mouse in mice:
        AP_source_coord_dicts = {}
        AP_position_dict = {}
        AP_position_vis_dict = {}
        vis_main_dict = {}
        barcodes = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded_with_source.pkl"
        )
        barcodes_no_soma = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded.pkl"
        )
        lcm_directory = proj_path / f"{mouse}/LCM"
        ROI_3D = np.load(lcm_directory / "ROI_3D_25.npy")
        all_VIS_ROI = np.unique(ROI_3D * VIS_mask * contra_mask)
        vis_coord = {}
        # to avoid A1 local projections influencing result, we remove VIS rois where more than 10% is in AUDp
        areas_only_grouped = get_area_volumes(
            barcode_table_cols=barcodes_no_soma.columns,
            lcm_directory=lcm_directory,
            area_threshold=0.1,
        )
        frac = areas_only_grouped.div(areas_only_grouped.sum(axis=1), axis=0)
        frac_filtered = frac.loc[
            (frac[HVA_cols].gt(0).any(axis=1)) & (frac["AUDp"] > 0.1)
        ].index
        all_VIS_ROI = [
            sample
            for sample in all_VIS_ROI
            if sample != 0
            and sample in barcodes_no_soma.columns
            and sample not in frac_filtered
            and areas_only_grouped[HVA_cols].loc[sample].sum() > 0
        ]

        for sample in all_VIS_ROI:
            centroid = np.argwhere(ROI_3D == sample).mean(axis=0)
            vis_coord[sample] = centroid
            AP_position_vis_dict[sample] = (
                max_y_vis - centroid[0]
            )  # centroid[0]-min_y_vis
            vis_main_dict[sample] = areas_only_grouped[HVA_cols].loc[sample].idxmax()
        all_AUDp_samples = np.unique(ROI_3D * AUDp_mask * contra_mask)
        all_AUDp_samples = [sample for sample in all_AUDp_samples if sample != 0]
        all_AUDp_samples = [
            sample for sample in all_AUDp_samples if sample in barcodes.columns
        ]
        for sample in all_AUDp_samples:
            centroid = np.argwhere(ROI_3D == sample).mean(axis=0)
            AP_position_dict[sample] = max_y_vis - centroid[0]  # -min_y#AP_midpoint_A1
            AP_source_coord_dicts[sample] = centroid
        mouse_dict_AP_source_coords[mouse] = AP_source_coord_dicts
        mouse_dict_AP_source[mouse] = AP_position_dict
        mouse_dict_AP_VC[mouse] = AP_position_vis_dict
        mouse_vis_main_dict[mouse] = vis_main_dict
        mouse_vis_coord[mouse] = vis_coord
    AP_positioning_dicts["mouse_dict_A1_source_coords"] = mouse_dict_AP_source_coords
    AP_positioning_dicts["mouse_dict_AP_source"] = mouse_dict_AP_source
    AP_positioning_dicts["mouse_dict_AP_VC"] = mouse_dict_AP_VC
    AP_positioning_dicts["mouse_vis_main_dict"] = mouse_vis_main_dict
    AP_positioning_dicts["mouse_vis_coord"] = mouse_vis_coord
    return AP_positioning_dicts


def get_A1_VC_centroid_coords():
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    AUDp_mask = get_area_mask(area_id=bg_atlas.structures["AUDp"]["id"])
    contra_mask = get_contra_mask(AUDp_mask.shape)
    VIS_mask = get_area_mask(bg_atlas.structures["VIS"]["id"])
    A1_masked = contra_mask * AUDp_mask
    A1_centroid_coords = np.argwhere(A1_masked == 1).mean(axis=0)
    vis_ipsi = VIS_mask * contra_mask
    VC_centroid_coords = np.argwhere(vis_ipsi == 1).mean(axis=0)
    return A1_centroid_coords, VC_centroid_coords


def get_AP_position(row, dictionary):
    key = row[0]
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return None


def get_mean_AP_soma_position(
    proj_path,
    mice,
    mouse_dict_AP_source,
    mouse_dict_AP_VC,
    mouse_vis_main_dict,
    mouse_vis_coord,
    max_y_vis,
    nearest_coord=True,
):
    """Function to get mean soma AP position for neurons targeting individual vis cubelets. We then create a dataframe for each vis cubelet, what is the mean AP position, what is the main
    visual area represented by that cubelet, what is the distance between that vis cubelet and A1 centre.
     Args:
        mice: list of mice ids used for finding paths
        proj_path: path to where pre-processed MAPseq datasets are stored
        mouse_dict_AP_source: (dict) for each mouse and each A1 source cubelet dict for their AP position
        mouse_dict_AP_VC: (dict) for each mouse vc cubelet, what is the AP position?
        mouse_vis_main_dict: (dict) for each mouse vc cubelet, what is the main VC area represented
        max_y_vis: most posterior coordinate in visual cortex (used to normalise A-P values)
    """
    AP_soma_VC_sample = pd.DataFrame(
        columns=[
            "Mouse",
            "mean_AP_soma",
            "AP_Vis",
            "VC_majority",
            "dist_3d",
            "sample",
        ]
    )
    AP_position_dict_list = {}
    for mouse in mice:
        bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
        A1_centroid_coords, VC_centroid_coords = get_A1_VC_centroid_coords()
        parameters_path = f"{proj_path}/{mouse}/Sequencing"
        barcodes = pd.read_pickle(
            f"{parameters_path}/A1_barcodes_thresholded_with_source.pkl"
        )
        barcodes = add_prefix_to_index(barcodes, mouse)
        soma = pd.DataFrame(barcodes.idxmax(axis=1))
        soma["AP_position"] = soma.apply(
            lambda row: get_AP_position(row, mouse_dict_AP_source[mouse]), axis=1
        )
        soma["mouse"] = mouse
        AUDp_mask = get_area_mask(area_id=bg_atlas.structures["AUDp"]["id"])
        contra_mask = get_contra_mask(AUDp_mask.shape)
        A1_masked = contra_mask * AUDp_mask
        # soma['uncorrected_AP']= soma.apply(lambda row: get_AP_position(row, mouse_dict_AP_source_uncorrected[mouse]), axis=1)
        for sample in mouse_dict_AP_VC[mouse].keys():
            indices_for_sample = barcodes[barcodes[sample] > 0].index
            if len(indices_for_sample) > 2:
                mean_AP = np.mean(soma.loc[indices_for_sample]["AP_position"])
                uncorrected_meanAP = -(mean_AP) + max_y_vis  # back to non-normalised

                A1_coord_updated = [
                    uncorrected_meanAP,
                    A1_centroid_coords[1],
                    A1_centroid_coords[2],
                ]
                vis_cubelet_coord_updated = [
                    mouse_vis_coord[mouse][sample][0],
                    VC_centroid_coords[1],
                    VC_centroid_coords[2],
                ]
                dist_3d = (
                    np.linalg.norm(
                        np.array(A1_coord_updated) - np.array(vis_cubelet_coord_updated)
                    )
                    * 25
                )
                new_row = pd.DataFrame(
                    {
                        "Mouse": [mouse],
                        "mean_AP_soma": [mean_AP * 25],
                        "AP_Vis": [mouse_dict_AP_VC[mouse][sample] * 25],
                        "VC_majority": [mouse_vis_main_dict[mouse][sample]],
                        "dist_3d": [dist_3d],
                        "sample": [sample],
                    }
                )
                AP_soma_VC_sample = pd.concat([AP_soma_VC_sample, new_row])
        AP_position_dict_list[mouse] = soma
    AP_position_dict_list_combined = pd.concat([AP_position_dict_list[k] for k in mice])
    return AP_position_dict_list_combined, AP_soma_VC_sample


def compute_mean_soma_AP_positions(gen_parameters):
    """Function to take paths in gen_parameters dict and process datasets of indicivual mice, taking mean soma AP position for each VC targeting cubelet.
    Returns a pandas dataframe for plotting
    """
    max_y, min_y, max_y_vis, min_y_vis, A1_centroid_coords = get_AUDp_VIS_coords()
    AP_positioning_dicts = get_AP_positioning_cubelets(
        mice=gen_parameters["MICE"],
        proj_path=gen_parameters["proj_path"],
        HVA_cols=gen_parameters["HVA_cols"],
        max_y_vis=max_y_vis,
    )
    AP_position_dict_list_combined, AP_soma_VC_sample = get_mean_AP_soma_position(
        proj_path=gen_parameters["proj_path"],
        mice=gen_parameters["MICE"],
        mouse_dict_AP_source=AP_positioning_dicts["mouse_dict_AP_source"],
        mouse_dict_AP_VC=AP_positioning_dicts["mouse_dict_AP_VC"],
        mouse_vis_main_dict=AP_positioning_dicts["mouse_vis_main_dict"],
        mouse_vis_coord=AP_positioning_dicts["mouse_vis_coord"],
        max_y_vis=max_y_vis,
    )
    return AP_position_dict_list_combined, AP_soma_VC_sample


def get_area_mean_AP(gen_parameters, combined_dict, AP_position_dict_list_combined):
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    mice = gen_parameters["MICE"]
    max_y, min_y, max_y_vis, min_y_vis, A1_centroid_coords = get_AUDp_VIS_coords()
    all_mice_combined = pd.concat(
        [
            combined_dict[k]["homogenous_across_cubelet"][
                get_common_columns(mice=mice, combined_dict=combined_dict, cortex=False)
            ]
            for k in mice
        ]
    )
    which_mice = pd.DataFrame(columns=["mice"], index=all_mice_combined.index)
    for k in mice:
        which_mice.loc[combined_dict[k]["homogenous_across_cubelet"].index, "mice"] = k

    area_AP_dict = {}
    where_AP_vis = {}
    for col in gen_parameters["HVA_cols"]:
        vals = []
        for mouse in mice:
            mouse_ind = which_mice[which_mice["mice"] == mouse].index
            mouse_bcs = all_mice_combined.loc[mouse_ind]
            proj_area = mouse_bcs[mouse_bcs[col] > 0]
            indices = proj_area.index
            if len(proj_area) > 0:
                AP_positions = AP_position_dict_list_combined.loc[indices][
                    "AP_position"
                ]
                vals.append(np.mean(AP_positions))
        AUDp_mask = get_area_mask(area_id=bg_atlas.structures["AUDp"]["id"])
        contra_mask = get_contra_mask(AUDp_mask.shape)
        VIS_area_mask = get_area_mask(bg_atlas.structures[col]["id"])
        VIS_area_mask = VIS_area_mask * contra_mask
        centroid = np.argwhere(VIS_area_mask == 1).mean(axis=0)
        where_AP_vis[col] = max_y_vis - centroid[0]
        area_AP_dict[col] = vals
    return where_AP_vis, area_AP_dict


def individual_area_probabilities(
    gen_parameters,
    combined_dict,
    AP_position_dict_list_combined,
    include_distance=False,
    distance_only=False,
):
    """function to use to assess the probabability of projecting to individual visual areas given neuron somas are in particular AP position.
    significance of relationship is assessed using logistic regression"""
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    AP_positioning_dicts = get_AP_positioning_cubelets(
        mice=gen_parameters["MICE"],
        proj_path=gen_parameters["proj_path"],
        HVA_cols=gen_parameters["HVA_cols"],
        max_y_vis=np.nan,
    )
    mice = gen_parameters["MICE"]
    all_mice_combined = pd.concat(
        [
            combined_dict[k]["homogenous_across_cubelet"][
                get_common_columns(mice=mice, combined_dict=combined_dict, cortex=False)
            ]
            for k in mice
        ]
    )
    which_mice = pd.DataFrame(columns=["mice"], index=all_mice_combined.index)
    for k in mice:
        which_mice.loc[combined_dict[k]["homogenous_across_cubelet"].index, "mice"] = k

    logistic_reg_area_dict = {}
    visual_areas = gen_parameters["HVA_cols"]
    for mouse in mice:
        mouse_ind = which_mice[which_mice["mice"] == mouse].index
        ap_corr_mouse = AP_position_dict_list_combined.loc[mouse_ind]["AP_position"]
        mouse_bcs = (
            all_mice_combined.loc[mouse_ind][visual_areas]
            .astype(bool)
            .astype(int)
            .copy()
        )
        for AP in ap_corr_mouse.unique():
            indices = ap_corr_mouse[ap_corr_mouse == AP]
            if len(indices) < 5:
                continue
            else:
                logistic_reg_area_dict[AP] = mouse_bcs.loc[indices.index]
                logistic_reg_area_dict[AP]["mouse"] = mouse
                logistic_reg_area_dict[AP]["AP_position"] = AP * 25
                soma = AP_position_dict_list_combined[
                    AP_position_dict_list_combined["AP_position"] == AP
                ][0].unique()[0]
                coordinates = AP_positioning_dicts["mouse_dict_A1_source_coords"][
                    mouse
                ][soma]
                logistic_reg_area_dict[AP]["source_sample_coord"] = [coordinates] * len(
                    logistic_reg_area_dict[AP]
                )

    combined_df = pd.concat(logistic_reg_area_dict.values(), axis=0, ignore_index=False)

    results_popuplation_dict = {}
    df = combined_df.melt(
        id_vars=["mouse", "AP_position", "source_sample_coord"],
        var_name="Area",
        value_name="Projection",
    )
    pval_df = pd.DataFrame(index=visual_areas, columns=["p_value", "OR"])
    AUDp_mask = get_area_mask(area_id=bg_atlas.structures["AUDp"]["id"])
    contra_mask = get_contra_mask(AUDp_mask.shape)

    for area in visual_areas:
        VIS_area_mask = get_area_mask(bg_atlas.structures[area]["id"])
        VIS_area_mask = VIS_area_mask * contra_mask
        centroid = np.argwhere(VIS_area_mask == 1).mean(axis=0)
        df.loc[df["Area"] == area, "distance"] = df.loc[
            df["Area"] == area, "source_sample_coord"
        ].apply(lambda coord: np.linalg.norm(coord - centroid) * 25)
        df_area = df[df["Area"] == area]
        variable_to_measure = "AP_position"
        if include_distance:
            model = smf.logit(
                "Projection ~ AP_position + mouse + distance", data=df_area
            ).fit(disp=False)
        elif not include_distance and not distance_only:
            model = smf.logit("Projection ~ AP_position + mouse", data=df_area).fit(
                disp=False
            )
        if distance_only:
            variable_to_measure = "distance"
            model = smf.logit("Projection ~ distance + mouse", data=df_area).fit(
                disp=False
            )
        results = model.summary2().tables[1]
        results_popuplation_dict[area] = smf.logit(
            f"Projection ~ {variable_to_measure}", data=df_area
        ).fit(disp=False)
        pval_df.loc[area, "p_value"] = results.loc[f"{variable_to_measure}", "P>|z|"]
        pval_df.loc[area, "OR"] = np.exp(results.loc[f"{variable_to_measure}", "Coef."])
    pval_df["p_value_corrected"] = pval_df["p_value"] * len(visual_areas)
    return pval_df, df, results_popuplation_dict


def pooled_area_probabilities(
    gen_parameters,
    combined_dict,
    AP_position_dict_list_combined,
    include_distance=True,  # keep distance as a covariate by default
):
    """
    We perform logistic regression on all areas at the same time, therefore distance is given the same weight across all areas (rather than separate weights if individual)
    """
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    AP_positioning_dicts = get_AP_positioning_cubelets(
        mice=gen_parameters["MICE"],
        proj_path=gen_parameters["proj_path"],
        HVA_cols=gen_parameters["HVA_cols"],
        max_y_vis=np.nan,
    )
    mice = gen_parameters["MICE"]
    visual_areas = gen_parameters["HVA_cols"]
    all_mice_combined = pd.concat(
        [
            combined_dict[k]["homogenous_across_cubelet"][
                get_common_columns(mice=mice, combined_dict=combined_dict, cortex=False)
            ]
            for k in mice
        ]
    )
    which_mice = pd.Series(
        {
            idx: k
            for k in mice
            for idx in combined_dict[k]["homogenous_across_cubelet"].index
        },
        name="mouse",
    ).to_frame()
    AUDp_mask = get_area_mask(area_id=bg_atlas.structures["AUDp"]["id"])
    contra_mask = get_contra_mask(AUDp_mask.shape)

    centroids = {}
    for area in visual_areas:
        mask = get_area_mask(bg_atlas.structures[area]["id"]) * contra_mask
        centroids[area] = np.argwhere(mask == 1).mean(axis=0)  # in voxels
    logistic_reg_area_dict = {}
    for mouse in mice:
        mouse_ind = which_mice[which_mice["mouse"] == mouse].index
        ap_corr_mouse = AP_position_dict_list_combined.loc[mouse_ind]["AP_position"]
        mouse_bcs = (
            all_mice_combined.loc[mouse_ind][visual_areas]
            .astype(bool)
            .astype(int)
            .copy()
        )

        for AP in ap_corr_mouse.unique():
            indices = ap_corr_mouse[ap_corr_mouse == AP]
            if len(indices) < 5:
                continue
            neurons = mouse_bcs.loc[indices.index].copy()
            neurons["mouse"] = mouse
            neurons["AP_position"] = AP * 25  # since 25um voxel resolution
            soma = AP_position_dict_list_combined[
                AP_position_dict_list_combined["AP_position"] == AP
            ][0].unique()[0]
            coord = AP_positioning_dicts["mouse_dict_A1_source_coords"][mouse][soma]
            neurons["source_sample_coord"] = [coord] * len(neurons)
            logistic_reg_area_dict[(mouse, AP)] = neurons

    combined_area_AP_proj_dict = pd.concat(
        logistic_reg_area_dict.values(), axis=0, ignore_index=False
    ).melt(
        id_vars=["mouse", "AP_position", "source_sample_coord"],
        var_name="Area",
        value_name="Projection",
    )
    combined_area_AP_proj_dict["distance"] = combined_area_AP_proj_dict.apply(
        lambda r: np.linalg.norm(r["source_sample_coord"] - centroids[r["Area"]]) * 25,
        axis=1,
    )

    base_formula = "Projection ~ AP_position * C(Area) + C(Area) + C(mouse)"
    if include_distance:
        base_formula += " + distance"

    model = smf.logit(base_formula, data=combined_area_AP_proj_dict).fit(disp=False)
    ref_area = combined_area_AP_proj_dict["Area"].unique().min()
    ref_AP_param = "AP_position"

    pval_df = pd.DataFrame(index=visual_areas, columns=["p_value", "OR"])

    for area in visual_areas:
        if area == ref_area:
            coef = model.params[ref_AP_param]
            pval = model.pvalues[ref_AP_param]
            var = model.cov_params().loc[ref_AP_param, ref_AP_param]
        else:
            inter_term = f"AP_position:C(Area)[T.{area}]"
            coef_vec = np.zeros(len(model.params))
            coef_vec[model.params.index.get_loc(ref_AP_param)] = 1
            coef_vec[model.params.index.get_loc(inter_term)] = 1
            # Wald test for the linear combination
            wald: ContrastResults = model.t_test(coef_vec)
            coef, pval, var = wald.effect[0], wald.pvalue, wald.sd**2

        pval_df.loc[area, "p_value"] = pval
        pval_df.loc[area, "OR"] = np.exp(coef)

    pval_df["p_value_corrected"] = pval_df["p_value"] * len(visual_areas)

    return pval_df, combined_area_AP_proj_dict, model


def get_ML_position(sample_id: str, ml_dict: dict):
    """return pre-computed ML coordinate for sample (or None)."""
    return ml_dict.get(sample_id, np.nan)


def analyse_AC_VC_ML_correlation(
    gen_parameters: dict,
    *,
    area_threshold: float = 0.1,  # max AUDp contamination allowed in VIS ROI
    min_audp_frac: float = 0.1,  # min AUDp frac in barcode to keep ROI
):
    """
    Fig. S4 (related to fig2.) - function to correlate medio-lateral (ML) somatic positions in AUDp with ML positions of projection ROIs in visual cortex (VC).
    """
    visual_areas = gen_parameters["HVA_cols"]
    proj_path = Path(gen_parameters["proj_path"])
    mice = gen_parameters["MICE"]
    mcc = MouseConnectivityCache()
    convert_dict = hf.get_convert_dict()
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    rsp = mcc.get_reference_space()
    AUDp_mask = get_area_mask(area_id=bg_atlas.structures["AUDp"]["id"])
    VIS_mask = get_area_mask(bg_atlas.structures["VIS"]["id"])
    contra_mask = get_contra_mask(AUDp_mask.shape)
    A1_masked = AUDp_mask * contra_mask
    # ML limits for normalisation
    indices_AUDp_contra = np.argwhere(A1_masked)
    min_x, max_x = indices_AUDp_contra[:, 2].min(), indices_AUDp_contra[:, 2].max()
    indices_VIS = np.argwhere(VIS_mask * contra_mask)
    min_x_vis, max_x_vis = indices_VIS[:, 2].min(), indices_VIS[:, 2].max()
    A1_centroid_coords = np.argwhere(A1_masked).mean(axis=0)
    ML_soma_VC_sample = pd.DataFrame(
        columns=["Mouse", "mean_ML_soma", "ML_Vis", "VC_majority", "dist_3d", "sample"]
    )
    ML_position_dict_list = {}
    for mouse in mice:
        ML_pos_audp = {}
        ML_pos_vis = {}
        vis_majority = {}
        vis_centroid = {}
        lcm_dir = proj_path / mouse / "LCM"
        ROI_3D = np.load(lcm_dir / "ROI_3D_25.npy")
        barcodes = pd.read_pickle(
            proj_path / mouse / "Sequencing" / "A1_barcodes_thresholded_with_source.pkl"
        )
        barcodes_no_soma = pd.read_pickle(
            proj_path / mouse / "Sequencing" / "A1_barcodes_thresholded.pkl"
        )
        areas_only_grouped = get_area_volumes(
            barcode_table_cols=barcodes_no_soma.columns,
            lcm_directory=lcm_dir,
            area_threshold=area_threshold,
        )
        frac = areas_only_grouped.div(areas_only_grouped.sum(axis=1), axis=0)
        frac_filtered = frac.loc[
            (frac[visual_areas].gt(0).any(axis=1)) & (frac["AUDp"] > min_audp_frac)
        ].index
        all_VIS_ROI = np.unique(ROI_3D * VIS_mask * contra_mask)
        all_VIS_ROI = [
            s
            for s in all_VIS_ROI
            if s != 0
            and s in barcodes_no_soma.columns
            and s not in frac_filtered
            and areas_only_grouped.loc[s, visual_areas].sum() > 0
        ]
        for sample in all_VIS_ROI:
            centroid = np.argwhere(ROI_3D == sample).mean(axis=0)
            vis_centroid[sample] = centroid
            ML_pos_vis[sample] = centroid[2] - min_x_vis
            vis_majority[sample] = areas_only_grouped.loc[sample, visual_areas].idxmax()
        all_AUDp_ROI = np.unique(ROI_3D * AUDp_mask * contra_mask)
        all_AUDp_ROI = [s for s in all_AUDp_ROI if s != 0 and s in barcodes.columns]
        for source_sample in all_AUDp_ROI:
            centroid = np.argwhere(ROI_3D == source_sample).mean(axis=0)
            ML_pos_audp[source_sample] = centroid[2] - min_x
        mouse_barcodes = add_prefix_to_index(barcodes, mouse)
        soma_df = pd.DataFrame(mouse_barcodes.idxmax(axis=1), columns=["sample"])
        soma_df["ML_pos"] = soma_df["sample"].map(lambda s: ML_pos_audp.get(s, np.nan))
        for vis_sample, ml_vis in ML_pos_vis.items():
            soma_indices = mouse_barcodes[mouse_barcodes[vis_sample] > 0].index
            if len(soma_indices) < 3:
                continue  # need ≥3 barcodes for stable mean
            mean_ML_norm = soma_df.loc[soma_indices, "ML_pos"].mean()
            mean_ML_abs = mean_ML_norm + min_x
            A1_coord = np.array(
                [A1_centroid_coords[0], A1_centroid_coords[1], mean_ML_abs]
            )
            dist_3d_um = np.linalg.norm(A1_coord - vis_centroid[vis_sample]) * 25
            ML_soma_VC_sample = pd.concat(
                [
                    ML_soma_VC_sample,
                    pd.DataFrame(
                        {
                            "Mouse": [mouse],
                            "mean_ML_soma": [mean_ML_norm * 25],
                            "ML_Vis": [ml_vis * 25],
                            "VC_majority": [vis_majority[vis_sample]],
                            "dist_3d": [dist_3d_um],
                            "sample": [vis_sample],
                        }
                    ),
                ]
            )

    ML_soma_VC_sample["converted"] = ML_soma_VC_sample["VC_majority"].map(convert_dict)

    rho, pval = pearsonr(
        ML_soma_VC_sample["ML_Vis"],
        ML_soma_VC_sample["mean_ML_soma"],
    )

    return (rho, pval, ML_soma_VC_sample)


def get_VIS_co_proj_distrib(visual_areas, all_mice):
    """function to perform analysis ins fig 2h to look at how many visual areas individual neurons project to"""
    vis_adj = [vis for vis in visual_areas if vis in all_mice.columns]
    vis_proj = all_mice[all_mice[vis_adj].astype(bool).sum(axis=1) > 0]
    vis_areas_per_neuron = vis_proj[vis_adj].astype(bool).sum(axis=1)
    return vis_areas_per_neuron


def get_max_counts(combined_dict, all_mice_combined, gen_parameters):
    max_counts_list = [data["max_counts"] for data in combined_dict.values()]
    vis_adj = [
        vis for vis in gen_parameters["HVA_cols"] if vis in all_mice_combined.columns
    ]
    vis_proj = all_mice_combined[
        all_mice_combined[vis_adj].astype(bool).sum(axis=1) > 0
    ]
    concatenated_max_counts = pd.concat(max_counts_list, ignore_index=False)
    max_counts = concatenated_max_counts.loc[vis_proj[vis_adj].index]
    return max_counts


def area_is_main(
    parameters_path,
    cortical,
    shuffled,
    barcode_matrix,
    IT_only=False,
    binary=False,
    run_externally=True,
):
    """
    Function to output a matrix of neuron barcode distribution across areas, where we assume that the main area in each cubelet is where the barcode counts belong to
    Args:
        parameters_path
        barcode_matrix = pandas dataframe with barcodes
        cortical (bool): True if you want onkly to look at cortical regions
        shuffled (bool): True if you want to shuffle values in all columns as a negative control
    """
    parameters = lf.load_parameters(directory=parameters_path)
    barcodes_across_sample = barcode_matrix

    if run_externally == False:
        sequencing_directory = Path(
            "".join(
                [
                    parameters["PROCESSED_DIR"],
                    "/",
                    parameters["PROJECT"],
                    "/",
                    parameters["MOUSE"],
                    "/Sequencing",
                ]
            )
        )
        processed_path = Path(PARAMETERS["data_root"]["processed"])
        lcm_directory = processed_path / (
            "turnerb_" + parameters["lcm_directory"].split("turnerb_")[1]
        )
    else:
        proj_path = parameters_path.split("/Sequencing")[0]
        lcm_directory = Path(f"{proj_path}/LCM")
    cortical_samples_columns = [
        int(col)
        for col in parameters["cortical_samples"]
        if col in barcodes_across_sample.columns
    ]
    # only look at cortical samples
    if cortical:
        barcodes_across_sample = barcodes_across_sample[cortical_samples_columns]
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample.astype(bool).sum(axis=1) > 0
    ]
    if IT_only:
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
        ]
    areas_only_grouped = get_area_volumes_as_main(
        barcode_table_cols=barcodes_across_sample.columns, lcm_directory=lcm_directory
    )
    areas_matrix = areas_only_grouped.to_numpy()
    total_frac = np.sum(areas_matrix, axis=1)
    frac_matrix = areas_matrix / total_frac[:, np.newaxis]
    weighted_frac_matrix = frac_matrix / areas_matrix.sum(axis=0)

    barcodes = barcodes_across_sample.to_numpy()
    if shuffled and not binary:
        barcodes = send_to_shuffle(barcodes=barcodes)
    if shuffled and binary:
        barcodes = send_for_curveball_shuff(barcodes=barcodes)
    total_projection_strength = np.sum(barcodes, axis=1)  # changed as normalised before
    # barcodes = barcodes / total_projection_strength[:, np.newaxis] #we don't normalise this anymore
    bc_matrix = np.matmul(barcodes, weighted_frac_matrix)
    bc_matrix = pd.DataFrame(
        data=bc_matrix,
        columns=areas_only_grouped.columns.to_list(),
        index=barcodes_across_sample.index,
    )
    bc_matrix = bc_matrix.dropna(axis=1, how="all")
    bc_matrix = bc_matrix.loc[:, (bc_matrix != 0).any(axis=0)]
    row_min = bc_matrix.min(axis=1)
    row_range = bc_matrix.max(axis=1) - row_min
    row_range.replace(0, np.nan, inplace=True)

    # bc_matrix = bc_matrix.sub(row_min, axis=0)
    bc_matrix = bc_matrix.div(row_range, axis=0)
    if binary:
        bc_matrix = bc_matrix.astype(bool).astype(int)
    return bc_matrix.fillna(0)


def homog_across_area(
    parameters_path,
    barcode_matrix,
    cortical,
    shuffled,
    IT_only=False,
    binary=False,
    run_externally=True,
):
    """
    Function to output a matrix of homogenous across areas, looking only at cortical samples
    Args:
        parameters_path
        barcode_matrix = pd.dataframe of barcodes
        binary (bool): True if you want to use the curveball algorithm for shuffling and actual data comparison, since requires the dataset to be binary
        cortical (bool): True if you want onkly to look at cortical regions
        shuffled (bool): True if you want to shuffle values in all columns as a negative control
        IT_only (bool): True if you want to look at only intratelencephalic neurons
    """
    parameters = lf.load_parameters(directory=parameters_path)
    barcodes_across_sample = barcode_matrix

    if run_externally == False:
        sequencing_directory = Path(
            "".join(
                [
                    parameters["PROCESSED_DIR"],
                    "/",
                    parameters["PROJECT"],
                    "/",
                    parameters["MOUSE"],
                    "/Sequencing",
                ]
            )
        )
        processed_path = Path(PARAMETERS["data_root"]["processed"])
        lcm_directory = processed_path / (
            "turnerb_" + parameters["lcm_directory"].split("turnerb_")[1]
        )
    else:
        proj_path = parameters_path.split("/Sequencing")[0]
        lcm_directory = Path(f"{proj_path}/LCM")
    cortical_samples_columns = [
        int(col)
        for col in parameters["cortical_samples"]
        if col in barcodes_across_sample.columns
    ]
    # only look at cortical samples
    if cortical:
        barcodes_across_sample = barcodes_across_sample[cortical_samples_columns]
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample.astype(bool).sum(axis=1) > 0
    ]
    if IT_only:
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
        ]
    areas_only_grouped = get_area_volumes(
        barcode_table_cols=barcodes_across_sample.columns, lcm_directory=lcm_directory
    )
    areas_matrix = areas_only_grouped.to_numpy()
    # total_frac = np.sum(areas_matrix, axis=1)
    # frac_matrix = areas_matrix/total_frac[:, np.newaxis]

    # barcodes_across_sample.fillna(0,inplace=True)
    barcodes_matrix = barcodes_across_sample.to_numpy()
    # barcodes_matrix[np.isnan(barcodes_matrix)] = 0
    if shuffled and not binary:
        barcodes_matrix = send_to_shuffle(barcodes=barcodes_matrix)

    if binary and shuffled:
        barcodes_matrix = send_for_curveball_shuff(barcodes=barcodes_matrix)
    total_projection_strength = np.sum(barcodes_matrix, axis=1)
    normalised_bc_matrix = barcodes_matrix / total_projection_strength[:, np.newaxis]
    normalised_bc_matrix = normalised_bc_matrix[
        total_projection_strength > 0, :
    ]  # needed as already removed barcodes with no projections but there are otherwise some nan values resulting from no projections in some barcodes after shuffling
    if not binary:
        mdl = Lasso(fit_intercept=False, positive=True)
        mdl.fit(areas_matrix, normalised_bc_matrix.T)
        barcodes_homog = pd.DataFrame(mdl.coef_, columns=areas_only_grouped.columns)
    if (
        binary
    ):  # if data is binarized, we will perform logistic regression rather than linear regression
        binarised_bc_matrix = (normalised_bc_matrix > 0).astype(int)
        # mdl = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=1000)
        mdl = LogisticRegression(
            penalty="l1", solver="saga", fit_intercept=False, max_iter=1000, C=1.0
        )
        coef_list = []
        for i in range(binarised_bc_matrix.shape[0]):
            mdl.fit(
                areas_matrix, binarised_bc_matrix[i, :]
            )  # fit logistic regression to each row (barcode)
            coef_list.append(mdl.coef_[0])  # store the coefficients for this barcode
        barcodes_homog = pd.DataFrame(coef_list, columns=areas_only_grouped.columns)
        barcodes_homog[
            barcodes_homog < 0
        ] = 0  # this is not ideal, but using sparse logistic regression, there isn't an option to contrain coef to be non-negative
        row_min = barcodes_homog.min(axis=1)
        row_range = barcodes_homog.max(axis=1) - row_min
        row_range.replace(0, np.nan, inplace=True)

        # barcodes_homog = barcodes_homog.sub(row_min, axis=0)
        barcodes_homog = barcodes_homog.div(
            row_range, axis=0
        )  # subtract min and divide by range so max = 1
        barcodes_homog = barcodes_homog.fillna(0)
    return barcodes_homog


def get_area_volumes_as_main(barcode_table_cols, lcm_directory):
    """
    Function to get volumes of each registered brain area from each LCM sample, but only the main area of each cubelet is kept
    Args:
        barcode_table_cols: list of column names of the barcode matrix
        lcm_directory: path to where the lcm_directory is
    Returns: area vol pandas dataframe
    """
    sample_vol_and_regions = pd.read_pickle(
        lcm_directory / "sample_vol_and_regions.pkl"
    )
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
        lambda row: row.where(row == row.max(), 0), axis=1
    )
    areas_only_grouped = areas_only_grouped.fillna(0)
    areas_only_grouped = areas_only_grouped.loc[
        :, (areas_only_grouped != 0).any(axis=0)
    ]
    return areas_only_grouped


def get_cond_prob(matrix, columns, index):
    """function to get the conditional probability a neuron projects to an area (column) given it projects to another area (index)"""
    conditional_prob = pd.DataFrame(
        data=np.zeros((len(index), len(columns))), columns=columns, index=index
    )
    matrix = matrix[columns]
    for col in index:
        for area in columns:
            # if col == area:
            #     conditional_prob.loc[col, area] = np.nan
            # else:
            conditional_prob.loc[col, area] = (
                matrix[matrix[col] > 0].astype(bool).astype(int)[area].mean()
            )
    return conditional_prob


def get_cosine_sim_of_probs(matrix, cols):
    cosine_sim_matrix = pd.DataFrame(
        data=np.zeros((len(cols), len(cols))), columns=cols, index=cols
    )
    for col in cols:
        for col_2 in cols:
            neurons_1 = matrix.loc[col]
            neurons_2 = matrix.loc[col_2]
            neurons_1 = neurons_1.drop([col, col_2])
            neurons_2 = neurons_2.drop([col, col_2])
            bl = np.array(neurons_1).reshape(1, -1)
            bl_2 = np.array(neurons_2).reshape(1, -1)
            cosine_sim = cosine_similarity(bl, bl_2)
            cosine_sim_matrix.loc[col, col_2] = cosine_sim[0][0]
            cosine_sim_matrix.loc[col_2, col] = cosine_sim[0][0]
    return cosine_sim_matrix


def get_p_val_comp_to_shuffled(proj_path, all_mice_combined, comp_VIS_only):
    """function to compute conditional probabilities and compare to values to the collated shuffled population to get a corrected p-val matrix"""
    conditional_probability_dict = {}
    hva_order_to_plot = [
        "VISl",
        "VISli",
        "VISpor",
        "VISpl",
        "VISp",
        "VISal",
        "VISam",
        "VISa",
        "VISpm",
        "VISrl",
    ]
    matrix = all_mice_combined.copy()
    all_cols = matrix.columns
    cols_reordered = [item for item in hva_order_to_plot if item in all_cols] + [
        item for item in all_cols if item not in hva_order_to_plot
    ]
    cols_reordered = [item for item in cols_reordered if item != "AUDp"]
    cols = [col for col in hva_order_to_plot if col in all_cols]
    matrix_to_comp = matrix.astype(bool).astype(int)
    conditional_prob = get_cond_prob(
        matrix=matrix_to_comp[cols_reordered],
        columns=cols_reordered,
        index=cols_reordered,
    )
    shuffled_cond_prob = pd.read_pickle(
        f"{proj_path}/collated_shuffles/shuffled_cubelet_conditional_prob__collated.pkl"
    )
    mean_val_matrix = pd.DataFrame(
        data=np.zeros((len(cols_reordered), len(cols_reordered))),
        columns=cols_reordered,
        index=cols_reordered,
    )
    p_val_matrix = pd.DataFrame(
        data=np.zeros((len(cols_reordered), len(cols_reordered))),
        columns=cols_reordered,
        index=cols_reordered,
    )
    for column_name in shuffled_cond_prob.columns:
        separated_words = column_name.split(", ")
        mean_corr = shuffled_cond_prob[column_name].mean()
        if (
            separated_words[0] in cols_reordered
            and separated_words[1] in cols_reordered
        ):
            mean_val_matrix.loc[separated_words[0], separated_words[1]] = mean_corr
            val_to_comp = conditional_prob.loc[separated_words[0], separated_words[1]]
            mu, std = norm.fit(shuffled_cond_prob[column_name].values)
            if val_to_comp >= mean_corr:
                p_val = norm.sf(val_to_comp, loc=mu, scale=std) * 2  # two-sided
            else:
                p_val = norm.cdf(val_to_comp, loc=mu, scale=std) * 2
            p_val_matrix.loc[separated_words[0], separated_words[1]] = p_val
    if comp_VIS_only:
        number_tests = len(cols) * (len(cols_reordered) - 1)
        mean_val_matrix = mean_val_matrix.loc[cols]
        conditional_prob = conditional_prob.loc[cols]
    else:
        number_tests = len(cols_reordered) * (len(cols_reordered) - 1)
    p_val_matrix_corrected = p_val_matrix * number_tests  # bonferroni correction
    conditional_probability_dict["shuffled"] = mean_val_matrix
    conditional_probability_dict["observed"] = conditional_prob
    return p_val_matrix_corrected, conditional_probability_dict


def perform_motif_analysis(gen_parameters, barcodes):
    """function to perform co-projection motif analysis in fig 2"""
    (
        shuffled_numbers,
        shuffled_2_combinations,
        shuffle_total_numbers,
    ) = lf.load_shuffled_matrices(proj_path=gen_parameters["proj_path"])
    cols = gen_parameters[
        "HVA_cols"
    ]  # [ "VISpl", "VISpor", "VISli", "VISl", "VISal", "VISrl", "VISa", "VISam", "VISpm",  "VISp", ]
    all_cols = barcodes.columns
    cols = [col for col in cols if col in all_cols]
    cols_reordered = [item for item in cols if item in all_cols] + [
        item for item in all_cols if item not in cols
    ]
    total_combinations = len(list(itertools.combinations(cols_reordered, 2)))
    cols_reordered = [item for item in cols_reordered if item != "AUDp"]
    probs_df = pd.DataFrame(
        index=["probs_actual", "probs_joint", "log_OR_actual", "mean_shuf", "p_value"]
    )
    for column_name in shuffled_2_combinations.columns:
        col, col_2 = map(str.strip, column_name.split(","))
        if col in all_cols and col_2 in all_cols:
            prob_df = pd.DataFrame()
            prob_df["a"] = barcodes[col].astype(bool)
            prob_df["b"] = barcodes[col_2].astype(bool)
            prob_df["matching"] = prob_df.apply(
                lambda x: 1 if x["a"] and x["b"] else 0, axis=1
            )
            probs_actual = prob_df["matching"].sum() / len(barcodes)
            probs_joint = (prob_df["a"].sum() / len(barcodes)) * (
                prob_df["b"].sum() / len(barcodes)
            )
            probs_df.loc["probs_actual", f"{col}, {col_2}"] = probs_actual
            probs_df.loc["probs_joint", f"{col}, {col_2}"] = probs_joint
            if f"{col}, {col_2}" in shuffled_2_combinations.columns:
                column_name = f"{col}, {col_2}"
            elif f"{col_2}, {col}" in shuffled_2_combinations.columns:
                column_name = f"{col_2}, {col}"
            shuff_actual_prob = (
                shuffled_2_combinations[column_name] / shuffle_total_numbers[0]
            )
            shuff_joint_prob = (shuffled_numbers[col] / shuffle_total_numbers[0]) * (
                shuffled_numbers[col_2] / shuffle_total_numbers[0]
            )
            shuff_effect = (shuff_actual_prob / shuff_joint_prob).astype(float)
            shuff_log_effect = np.log2(shuff_effect + 1e-3)
            mean_shuff_log_effect = shuff_log_effect.mean()
            actual_effect = probs_actual / probs_joint
            probs_df.loc["mean_shuf", f"{col}, {col_2}"] = mean_shuff_log_effect
            probs_df.loc["log_actual", f"{col}, {col_2}"] = np.log2(
                actual_effect + 1e-3
            )
            mu, std = norm.fit(shuff_effect)
            if actual_effect >= shuff_effect.mean():
                p_val = norm.sf(actual_effect, loc=mu, scale=std) * 2  # two-sided
            else:
                p_val = norm.cdf(actual_effect, loc=mu, scale=std) * 2
            p_val_adj = p_val * total_combinations
            probs_df.loc["p_value", f"{col}, {col_2}"] = p_val_adj
    to_plot = probs_df.T
    to_plot["-log10_p_value"] = -np.log10(to_plot["p_value"])
    to_plot["shuf-sub"] = to_plot["log_actual"] - to_plot["mean_shuf"]
    return to_plot


def cal_cosine_sim(gen_parameters, all_mice_combined):
    """calc cosine simularity in conditional probability"""
    proj_path = gen_parameters["proj_path"]
    pval_adj, cp_dict = get_p_val_comp_to_shuffled(
        proj_path=proj_path, all_mice_combined=all_mice_combined, comp_VIS_only=True
    )
    cols = gen_parameters["HVA_cols"]
    all_cols = all_mice_combined.columns
    not_in = [col for col in all_cols if col != "AUDp" and col not in cols]
    combined = cols + not_in
    cosine_df = get_cosine_sim_of_probs(
        cp_dict["observed"].loc[cols][combined], cols=cols
    )
    return cosine_df


def get_dict_area_vols(proj_path, mice):
    area_dictionary = {}
    for num, mouse in enumerate(mice):
        new_dict = {}
        barcodes = pd.read_pickle(
            f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded.pkl"
        )
        lcm_directory = Path(f"{proj_path}/{mouse}/LCM")
        new_dict["random_key"] = get_area_volumes(
            barcode_table_cols=barcodes.columns, lcm_directory=lcm_directory
        )
        area_dictionary[mouse] = new_dict
    return area_dictionary


def get_area_sample_corr(area_dictionary, mice):
    common_cols_cortex = get_common_columns(
        mice=mice, combined_dict=area_dictionary, key="random_key", cortex=True
    )
    corr_dict = {}
    for mouse in mice:
        common_area_mouse_mat = area_dictionary[mouse]["random_key"][common_cols_cortex]
        corr_dict[mouse] = common_area_mouse_mat.corr(method="spearman")
    return corr_dict


def compute_actual_expected_ratio_matrix(df):
    cols = df.columns
    ratio_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col1, col2 in itertools.combinations(cols, 2):
        a = df[col1].astype(bool)
        b = df[col2].astype(bool)
        actual = (a & b).sum() / len(df)
        expected = (a.sum() / len(df)) * (b.sum() / len(df))
        ratio = actual / expected if expected > 0 else float("nan")
        ratio_matrix.loc[col1, col2] = ratio
        ratio_matrix.loc[col2, col1] = ratio
    np.fill_diagonal(ratio_matrix.values, np.nan)
    return ratio_matrix


def compare_shuffle_approaches(mice, proj_path, all_mice_combined):
    """function to compare different shuffle approaches - one using the curveball algorithm to account for variable labelling efficiency"""
    shuf_combined_dict = samples_to_areas(
        mice=mice, proj_path=proj_path, shuffled=True, binary=False
    )
    shuf_all_mice_combined = pd.concat(
        [
            shuf_combined_dict[k]["homogenous_across_cubelet"][
                get_common_columns(
                    mice=mice, combined_dict=shuf_combined_dict, cortex=True
                )
            ]
            for k in mice
        ]
    )
    curve_shuf_combined_dict = samples_to_areas(
        mice=mice, proj_path=proj_path, shuffled=True, binary=True
    )
    curve_shuf_all_mice_combined = pd.concat(
        [
            curve_shuf_combined_dict[k]["homogenous_across_cubelet"][
                get_common_columns(
                    mice=mice, combined_dict=curve_shuf_combined_dict, cortex=True
                )
            ]
            for k in mice
        ]
    )
    titles = ["observed", "shuffled", "curveball_shuffled"]
    effect_dict = {}
    for i, matrix in enumerate(
        [all_mice_combined, shuf_all_mice_combined, curve_shuf_all_mice_combined]
    ):
        effect_dict[titles[i]] = compute_actual_expected_ratio_matrix(matrix)
    observed_over_shuff = {}
    observed_over_shuff["norm_shuff"] = (np.log2(effect_dict["observed"] + 1e-3)) - (
        np.log2(effect_dict["shuffled"] + 1e-3)
    )
    observed_over_shuff["curveball_shuffled"] = (
        np.log2(effect_dict["observed"] + 1e-3)
    ) - (np.log2(effect_dict["curveball_shuffled"] + 1e-3))
    return observed_over_shuff


def perform_shuffle_motif_comp(neurons_proj, n_areas=10):
    n_neurons = len(neurons_proj)
    norm_shuff_barcodes = send_to_shuffle(barcodes=neurons_proj.astype(int))
    curveball_shuff_barcodes = send_for_curveball_shuff(
        barcodes=neurons_proj.astype(int)
    )
    barcode_dict = {}
    titles = ["actual", "norm_shuffled", "curvball_shuffled"]
    for i, barcode_df in enumerate(
        [neurons_proj.astype(int), norm_shuff_barcodes, curveball_shuff_barcodes]
    ):
        motif_df = pd.DataFrame(np.zeros((n_areas, n_areas)))
        for area_a in range(n_areas):
            for area_b in range(n_areas):
                if area_a == area_b:
                    continue
                observed = (
                    np.count_nonzero(barcode_df[:, area_a] & barcode_df[:, area_b])
                ) / n_neurons
                expected = ((np.count_nonzero(barcode_df[:, area_a])) / n_neurons) * (
                    (np.count_nonzero(barcode_df[:, area_b])) / n_neurons
                )
                motif_df.iloc[area_a, area_b] = np.log2((observed / expected) + 1e-3)
        barcode_dict[titles[i]] = motif_df
    norm_shuf_sub = barcode_dict["actual"] - barcode_dict["norm_shuffled"]
    curve_shuf_sub = barcode_dict["actual"] - barcode_dict["curvball_shuffled"]
    return norm_shuf_sub, curve_shuf_sub
