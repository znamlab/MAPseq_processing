from bg_atlasapi import BrainGlobeAtlas

# from preprocessing_sequencing import preprocess_sequences as ps
from znamutils import slurm_it
import pandas as pd
from final_processing import final_processing_functions as fpf
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
from bg_atlasapi import BrainGlobeAtlas
import statsmodels.formula.api as smf

def samples_to_areas(mice, proj_path):
    """function to generate a dictionary of with mice as keys for neuron barcodes across areas (from neuron barcodes across samples) """
    combined_dict = {}
    for num, mouse in enumerate(mice):
        new_dict = {}
        parameters_path = (
        f"{proj_path}/{mouse}/Sequencing")
        barcodes = pd.read_pickle(f"{parameters_path}/A1_barcodes_thresholded.pkl")
        barcodes = add_prefix_to_index(barcodes, mouse)
        new_dict['homogenous_across_cubelet'] = homog_across_cubelet(parameters_path=parameters_path, barcode_matrix = barcodes, cortical=True, IT_only=True, shuffled=False)
        new_dict['max_counts'] = barcodes.max(axis=1)
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
    binary=False, remove_AUDp_vis_cub=False, run_externally = False
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
        proj_path = parameters_path.split('/Sequencing')[0]
        lcm_directory = Path(f'{proj_path}/LCM')
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
    if remove_AUDp_vis_cub ==True:
        visual_areas = ['VISli','VISpor', 'VISpl', 'VISl', 'VISp', 'VISal', 'VISam', 'VISpm', 'VISa', 'VISrl']
        frac = areas_only_grouped.div(areas_only_grouped.sum(axis=1), axis=0)
        frac_filtered = frac.loc[(frac[visual_areas].gt(0).any(axis=1)) & (frac['AUDp'] > 0.1)].index
        barcodes_across_sample.drop(columns=frac_filtered, inplace=True)
        areas_only_grouped = areas_only_grouped.drop(index=frac_filtered, errors='ignore') 
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
    row_min    = bc_matrix.min(axis=1)                         
    row_range  = bc_matrix.max(axis=1) - row_min               
    row_range.replace(0, np.nan, inplace=True)                

    bc_matrix  = bc_matrix.sub(row_min,   axis=0)             
    bc_matrix  = bc_matrix.div(row_range, axis=0)         

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

def get_common_columns(mice, combined_dict, cortex, key= 'homogenous_across_cubelet'):
    """Function to get common areas across mouse barcode dictionaries. If cortex ==True, only take cortical areas"""
    mcc = MouseConnectivityCache(resolution=25)
    structure_tree = mcc.get_structure_tree()
    common_columns = set.intersection(*[
    set(combined_dict[k][key].columns) for k in mice
])
    #let's make sure that all the areas are cortical (areas such as HPF are unintentially side bits of cubelets and never main target, and more likely registration errors)
    if cortex:
        common_cols_cortex = []
        for col in common_columns:
            if col == 'Contra':
                common_cols_cortex.append(col)
            if col not in ['Contra', 'OB']:
                structure = structure_tree.get_structures_by_acronym([col])
                if 315 in structure[0]['structure_id_path']:
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
    return pd.to_numeric(df.mean(), errors='coerce'), pd.to_numeric(df.std(), errors='coerce')

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
        mask = np.ones(len(distances), dtype=bool); mask[i] = False
        params_i, _ = curve_fit(exponential, distances[mask], means[mask], p0=p0)
        loo_preds.iloc[i] = exponential(distances.iloc[i], *params_i)
    cv_corr, cv_p = pearsonr(means, loo_preds)

    return FitResult(params, ks_stat, ks_p, cv_corr, cv_p, fitted_x, fitted_y)

def get_distances_from_A1(combined_dict, area_cols, mice):
    """function to make a dataframe of distances from A1"""
    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()
    rsp = mcc.get_reference_space()
    #a1_dist_dict = {}
    structure = structure_tree.get_structures_by_acronym(['AUDp'])
    structure_id = structure[0]['id']
    mask = rsp.make_structure_mask([structure_id], direct_only=False)
    A1_coord = (np.mean(np.where(mask == 1)[0]), np.mean(np.where(mask == 1)[1]), np.mean(np.where(mask == 1)[2]))
    key_to_plot = 'homogenous_across_cubelet'
    areas = area_cols.drop(['Contra', 'AUDp'])
    # vis_adj = [vis for vis in visual_areas if vis in all_mice[key_to_plot].columns]
    distance_from_a1 = pd.DataFrame(index=areas, columns=['dist'])
    for col in areas:
        structure = structure_tree.get_structures_by_acronym([col])
        structure_id = structure[0]['id']
        mask = rsp.make_structure_mask([structure_id], direct_only=False)
        vis_coord = np.mean(np.where(mask == 1)[0]), np.mean(np.where(mask == 1)[1]), np.mean(np.where(mask == 1)[2])
        distance_from_a1.loc[col] = np.linalg.norm(np.array(A1_coord) - np.array(vis_coord)) * 25

    #a1_dist_dict[key_to_plot] = distance_from_a1
    freq_df = pd.DataFrame(columns=areas, index=mice)
    freq_df_strength = pd.DataFrame(columns=areas, index=mice)
    for mouse in mice:
        freq_df.loc[mouse] = combined_dict[mouse][key_to_plot][areas].astype(bool).sum(axis=0) / len(combined_dict[mouse][key_to_plot])
        freq_df_strength.loc[mouse] = combined_dict[mouse][key_to_plot][areas].where(combined_dict[mouse][key_to_plot][areas] > 0).mean(axis=0)
    distances = pd.Series(distance_from_a1.iloc[:, 0], index=areas)
    distances = pd.to_numeric(distances, errors='coerce')
    return freq_df, freq_df_strength, distances

def get_contra_mask(mask_shape):
    contra = np.zeros(mask_shape, dtype=bool)
    contra[:, :, mask_shape[2] // 2:] = 1
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
    #mcc = MouseConnectivityCache()
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    # AUDp_id = bg_atlas.structures['AUDp']['id']
    # VIS_id = bg_atlas.structures['VIS']['id']
    #rsp = mcc.get_reference_space()
    AUDp_mask = get_area_mask(area_id=bg_atlas.structures['AUDp']['id'])#rsp.make_structure_mask([AUDp_id], direct_only=False)
    indices_AUDp = np.argwhere(AUDp_mask == 1)
    VIS_mask = get_area_mask(bg_atlas.structures['VIS']['id'])#area_id=669) #rsp.make_structure_mask([669], direct_only=False) #669 is id for whole visual cortex
    indices_VIS = np.argwhere(VIS_mask == 1)
    max_y_vis = np.max(indices_VIS[:, 0])
    min_y_vis = np.min(indices_VIS[:, 0])
    #select anterior and posterior parts of A1
    max_y = np.max(indices_AUDp[:, 0])
    min_y = np.min(indices_AUDp[:, 0])
    # AP_midpoint_A1 = ((max_y - min_y) /2) + min_y
    # posterior_neurons = indices_AUDp[indices_AUDp[:, 0]>=AP_midpoint_A1]
    # anterior_neurons = indices_AUDp[indices_AUDp[:, 0]<AP_midpoint_A1]
    #now select only the ipsiliateral side of where was injected
    # x_midpoint = AUDp_mask.shape[2] // 2
    contra_mask = get_contra_mask(AUDp_mask.shape)
    # contra_mask = np.zeros_like(AUDp_mask, dtype=bool)
    # contra_mask[:, :, x_midpoint:] = 1
    #lets get the coordinates for the centre of A1
    A1_masked = contra_mask * AUDp_mask
    A1_centroid_coords = np.argwhere(A1_masked == 1).mean(axis=0)
    return max_y, min_y, max_y_vis, min_y_vis, A1_centroid_coords

def get_AP_positioning_cubelets(mice, proj_path ,HVA_cols, max_y_vis):
    """Function to create dictionaries for A-P coordinate positions of A1 soma cubelets and visual cortex target cubelets
    Args:
        mice: list of mice ids used for finding paths
        proj_path: path to where pre-processed MAPseq datasets are stored
        HVA_cols: list of visual cortex areas we use to limit analysis to cubelets in these areas
        max_y_vis: most posterior coordinate in visual cortex (used to normalise A-P values)"""
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    proj_path = Path(proj_path)
    mouse_dict_AP_source = {}
    mouse_dict_AP_VC = {}
    # mouse_barcodes_by_source = {}
    mouse_vis_main_dict = {}
    mouse_vis_coord = {}
    AP_positioning_dicts = {}
    AUDp_mask = get_area_mask(area_id=bg_atlas.structures['AUDp']['id'])
    contra_mask = get_contra_mask(AUDp_mask.shape)
    VIS_mask = get_area_mask(bg_atlas.structures['VIS']['id'])
    for mouse in mice:
        #if mouse == 'FIAA45.6d':
        AP_position_dict = {}
        AP_position_vis_dict = {}
        vis_main_dict = {}
        barcodes = pd.read_pickle(f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded_with_source.pkl")
        barcodes_no_soma = pd.read_pickle(f"{proj_path}/{mouse}/Sequencing/A1_barcodes_thresholded.pkl")
        lcm_directory = proj_path/f"{mouse}/LCM"
        ROI_3D = np.load(lcm_directory / "ROI_3D_25.npy")
        all_VIS_ROI = np.unique(ROI_3D *  VIS_mask * contra_mask)
        vis_coord = {}
        #to avoid A1 local projections influencing result, we remove VIS rois where more than 10% is in AUDp
        areas_only_grouped = fpf.get_area_volumes(
            barcode_table_cols=barcodes_no_soma.columns,
            lcm_directory=lcm_directory,
            area_threshold=0.1,
        )
        frac = areas_only_grouped.div(areas_only_grouped.sum(axis=1), axis=0)
        frac_filtered = frac.loc[(frac[HVA_cols].gt(0).any(axis=1)) & (frac['AUDp'] > 0.1)].index
        all_VIS_ROI = [sample for sample in all_VIS_ROI if sample != 0 and sample in barcodes_no_soma.columns and sample not in frac_filtered and areas_only_grouped[HVA_cols].loc[sample].sum()>0]

        for sample in all_VIS_ROI:
            centroid = np.argwhere(ROI_3D == sample).mean(axis=0)
            vis_coord[sample] = centroid
            AP_position_vis_dict[sample] = max_y_vis - centroid[0] #centroid[0]-min_y_vis
            vis_main_dict[sample] = areas_only_grouped[HVA_cols].loc[sample].idxmax()
        all_AUDp_samples = np.unique(ROI_3D *  AUDp_mask * contra_mask)
        all_AUDp_samples = [sample for sample in all_AUDp_samples if sample != 0]
        all_AUDp_samples = [sample for sample in all_AUDp_samples if sample in barcodes.columns]
        for sample in all_AUDp_samples:
            centroid = np.argwhere(ROI_3D == sample).mean(axis=0)
            AP_position_dict[sample] = max_y_vis - centroid[0]#-min_y#AP_midpoint_A1
        mouse_dict_AP_source[mouse]=AP_position_dict
        mouse_dict_AP_VC[mouse] = AP_position_vis_dict
        mouse_vis_main_dict[mouse] = vis_main_dict
        mouse_vis_coord[mouse] = vis_coord
    AP_positioning_dicts['mouse_dict_AP_source'] = mouse_dict_AP_source
    AP_positioning_dicts['mouse_dict_AP_VC'] = mouse_dict_AP_VC
    AP_positioning_dicts['mouse_vis_main_dict'] = mouse_vis_main_dict
    AP_positioning_dicts['mouse_vis_coord'] = mouse_vis_coord
    return AP_positioning_dicts

def get_A1_VC_centroid_coords():
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    AUDp_mask = get_area_mask(area_id=bg_atlas.structures['AUDp']['id'])
    contra_mask = get_contra_mask(AUDp_mask.shape)
    VIS_mask = get_area_mask(bg_atlas.structures['VIS']['id'])
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

def get_mean_AP_soma_position(proj_path, mice, mouse_dict_AP_source, mouse_dict_AP_VC, mouse_vis_main_dict, mouse_vis_coord, max_y_vis):
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
    AP_soma_VC_sample = pd.DataFrame(columns=['Mouse', 'mean_AP_soma', 'AP_Vis', 'VC_majority', 'dist_3d', 'sample'])
    AP_position_dict_list = {}
    for mouse in mice:
        A1_centroid_coords, VC_centroid_coords = get_A1_VC_centroid_coords()
        parameters_path = (
        f"{proj_path}/{mouse}/Sequencing")
        barcodes = pd.read_pickle(f"{parameters_path}/A1_barcodes_thresholded_with_source.pkl")
        barcodes= fpf.add_prefix_to_index(barcodes, mouse)
        soma = pd.DataFrame(barcodes.idxmax(axis=1))
        soma['AP_position'] = soma.apply(lambda row: get_AP_position(row, mouse_dict_AP_source[mouse]), axis=1)
        soma['mouse'] = mouse
        #soma['uncorrected_AP']= soma.apply(lambda row: get_AP_position(row, mouse_dict_AP_source_uncorrected[mouse]), axis=1)
        for sample in mouse_dict_AP_VC[mouse].keys():
            indices_for_sample = barcodes[barcodes[sample]>0].index
            if len(indices_for_sample)>2:
                mean_AP=np.mean(soma.loc[indices_for_sample]['AP_position'])
                uncorrected_meanAP = -(mean_AP) + max_y_vis #back to non-normalised
                A1_coord_updated = [uncorrected_meanAP, A1_centroid_coords[1], A1_centroid_coords[2]]
                vis_cubelet_coord_updated = [mouse_vis_coord[mouse][sample][0], VC_centroid_coords[1], VC_centroid_coords[2]]
                dist_3d = np.linalg.norm(np.array(A1_coord_updated) - np.array(vis_cubelet_coord_updated)) * 25
                new_row= pd.DataFrame({'Mouse':[mouse], 'mean_AP_soma':[mean_AP*25], 'AP_Vis':[mouse_dict_AP_VC[mouse][sample]*25], 'VC_majority': [mouse_vis_main_dict[mouse][sample]], 'dist_3d': [dist_3d], 'sample': [sample]})
                AP_soma_VC_sample = pd.concat([AP_soma_VC_sample, new_row])
        AP_position_dict_list[mouse] = soma
    AP_position_dict_list_combined = pd.concat([
        AP_position_dict_list[k] for k in mice
    ])
    return AP_position_dict_list_combined, AP_soma_VC_sample

def compute_mean_soma_AP_positions(gen_parameters):
    """Function to take paths in gen_parameters dict and process datasets of indicivual mice, taking mean soma AP position for each VC targeting cubelet.
    Returns a pandas dataframe for plotting
    """
    max_y, min_y, max_y_vis, min_y_vis, A1_centroid_coords = get_AUDp_VIS_coords()
    AP_positioning_dicts = get_AP_positioning_cubelets(mice=gen_parameters['MICE'], proj_path=gen_parameters['proj_path'], HVA_cols=gen_parameters['HVA_cols'], max_y_vis= max_y_vis)
    AP_position_dict_list_combined, AP_soma_VC_sample =  get_mean_AP_soma_position(proj_path=gen_parameters['proj_path'], mice=gen_parameters['MICE'], mouse_dict_AP_source=AP_positioning_dicts['mouse_dict_AP_source'], mouse_dict_AP_VC= AP_positioning_dicts['mouse_dict_AP_VC'], mouse_vis_main_dict=AP_positioning_dicts['mouse_vis_main_dict'], mouse_vis_coord= AP_positioning_dicts['mouse_vis_coord'], max_y_vis=max_y_vis)
    return AP_position_dict_list_combined, AP_soma_VC_sample

def get_area_mean_AP(gen_parameters, combined_dict, AP_position_dict_list_combined):
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    mice = gen_parameters['MICE']
    max_y, min_y, max_y_vis, min_y_vis, A1_centroid_coords = get_AUDp_VIS_coords()
    all_mice_combined = pd.concat([
    combined_dict[k]['homogenous_across_cubelet'][get_common_columns(mice=mice, combined_dict=combined_dict, cortex=False)]
    for k in mice
])
    which_mice = pd.DataFrame(columns = ['mice'], index= all_mice_combined.index)
    for k in mice:
        which_mice.loc[combined_dict[k]['homogenous_across_cubelet'].index, 'mice'] = k
        
    area_AP_dict = {}
    where_AP_vis = {}
    for col in gen_parameters['HVA_cols']:
        vals = []
        for mouse in mice:
            mouse_ind = which_mice[which_mice['mice']==mouse].index
            mouse_bcs = all_mice_combined.loc[mouse_ind]
            proj_area = mouse_bcs[mouse_bcs[col]>0]
            indices = proj_area.index
            if len(proj_area)>0:
                AP_positions = AP_position_dict_list_combined.loc[indices]['AP_position']
                vals.append(np.mean(AP_positions))
        AUDp_mask = get_area_mask(area_id=bg_atlas.structures['AUDp']['id'])
        contra_mask = get_contra_mask(AUDp_mask.shape)
        VIS_area_mask = get_area_mask(bg_atlas.structures[col]['id'])
        VIS_area_mask = VIS_area_mask* contra_mask
        centroid = np.argwhere(VIS_area_mask == 1).mean(axis=0)
        where_AP_vis[col] = max_y_vis - centroid[0]
        area_AP_dict[col] = vals
    return where_AP_vis, area_AP_dict

def individual_area_probabilities(gen_parameters, combined_dict, AP_position_dict_list_combined):
    """function to use to assess the probabability of projecting to individual visual areas given neuron somas are in particular AP position. 
     significance of relationship is assessed using logistic regression """
    mice = gen_parameters['MICE']
    area_dic= {}
    all_mice_combined = pd.concat([
    combined_dict[k]['homogenous_across_cubelet'][get_common_columns(mice=mice, combined_dict=combined_dict, cortex=False)]
    for k in mice
])
    for area in all_mice_combined.columns:
        which_mice = pd.DataFrame(columns = ['mice'], index= all_mice_combined.index)
    for k in mice:
        which_mice.loc[combined_dict[k]['homogenous_across_cubelet'].index, 'mice'] = k
        area_df =pd.DataFrame(columns=['mouse', 'AP_position', 'proj_freq'])
        for mouse in mice:
            mouse_ind = which_mice[which_mice['mice']==mouse].index
            ap_corr_mouse = AP_position_dict_list_combined.loc[mouse_ind]['AP_position']
            mouse_bcs = all_mice_combined.loc[mouse_ind].astype(bool).astype(int)
            for AP in ap_corr_mouse.unique():
                indices = ap_corr_mouse[ap_corr_mouse==AP]
                if len(indices)<5:
                    continue
                else:
                    freq = mouse_bcs.loc[indices.index][area].mean()
                    new_row = pd.DataFrame({"mouse": [mouse], "AP_position": [AP*25], "proj_freq": [freq]})
                    area_df = pd.concat([area_df, new_row], ignore_index=True)
            area_dic[area] = area_df

    logistic_reg_area_dict = {}
    visual_areas = gen_parameters['HVA_cols']
    for mouse in mice:
        mouse_ind = which_mice[which_mice['mice']==mouse].index
        ap_corr_mouse = AP_position_dict_list_combined.loc[mouse_ind]['AP_position']
        mouse_bcs = all_mice_combined.loc[mouse_ind][visual_areas].astype(bool).astype(int).copy()
        for AP in ap_corr_mouse.unique():
            indices = ap_corr_mouse[ap_corr_mouse==AP]
            if len(indices)<5:
                continue
            else:
                logistic_reg_area_dict[AP] = mouse_bcs.loc[indices.index]
                logistic_reg_area_dict[AP]['mouse'] = mouse
                logistic_reg_area_dict[AP]['AP_position'] = AP*25

    combined_df = pd.concat(logistic_reg_area_dict.values(), axis=0, ignore_index=False)

    results_popuplation_dict = {}
    df = combined_df.melt(id_vars=['mouse', 'AP_position'], var_name='Area', value_name='Projection')
    pval_df = pd.DataFrame(index=visual_areas, columns=['p_value', 'OR'])
    for area in visual_areas:
        df_area = df[df['Area'] == area]
        model = smf.logit('Projection ~ AP_position + mouse', data=df_area).fit(disp=False)
        results = model.summary2().tables[1]
        results_popuplation_dict[area] = smf.logit('Projection ~ AP_position', data=df_area).fit(disp=False)
        pval_df.loc[area, 'p_value'] = results.loc['AP_position', 'P>|z|']
        pval_df.loc[area, 'OR'] = np.exp(results.loc['AP_position', 'Coef.'])
    pval_df['p_value_corrected'] = pval_df['p_value']*len(visual_areas)
    return pval_df, df, results_popuplation_dict