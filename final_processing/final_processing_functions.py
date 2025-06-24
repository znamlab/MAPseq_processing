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
from flexiznam.config import PARAMETERS
import subprocess


def find_adjacent_samples(ROI_array, samples_to_look, parameters_path):
    """
    Function to find adjacent samples surrounding cubelets within 25um distance of max
    Args:
        ROI_array: 3D numpy array in 25um resolution
        samples_to_look: list of samples you want to find adjacent samples for
    Returns:
        Dictionary containing adjacent samples for each sample
    """
    voxels_to_extend = 1  # with 25um resolution, you're scanning 25um either end
    adjacent_dict = {}
    parameters = load_parameters(directory=parameters_path)
    cortical_samples = parameters["cortical_samples"]
    for ROI_sample in samples_to_look:
        coordinates = np.argwhere(ROI_array == ROI_sample)
        sample_list = []
        for coord in coordinates:
            # for each coordinate in ROI extend and subtract in each axis to find samples in close to the sample you're interested in
            sample_num = ROI_array[coord[0] + voxels_to_extend, coord[1], coord[2]]
            if sample_num != 0 and sample_num != ROI_sample:
                sample_list.append(sample_num)
            sample_num = ROI_array[coord[0] - voxels_to_extend, coord[1], coord[2]]
            if sample_num != 0 and sample_num != ROI_sample:
                sample_list.append(sample_num)
            sample_num = ROI_array[coord[0], coord[1] + voxels_to_extend, coord[2]]
            if sample_num != 0 and sample_num != ROI_sample:
                sample_list.append(sample_num)
            sample_num = ROI_array[coord[0], coord[1] - voxels_to_extend, coord[2]]
            if sample_num != 0 and sample_num != ROI_sample:
                sample_list.append(sample_num)
            sample_num = ROI_array[coord[0], coord[1], coord[2] + voxels_to_extend]
            if sample_num != 0 and sample_num != ROI_sample:
                sample_list.append(sample_num)
            sample_num = ROI_array[coord[0], coord[1], coord[2] - voxels_to_extend]
            if sample_num != 0 and sample_num != ROI_sample:
                sample_list.append(sample_num)
        adjacent_samples = np.unique(sample_list)
        adjacent_dict[ROI_sample] = [
            sample for sample in adjacent_samples if sample in cortical_samples
        ]  # since dendrites and split somas will be in cortical samples, don't include adjacent samples as non-cortical
    return adjacent_dict


def rename_tubes(barcode_table, parameters_path):
    """
    Function to rename RT primer names with sample tube names
    Args:
        barcode_table: pandas dataframe containing normalised barcode matrix
        parameters_path: path to where parameters yaml file is
    """
    # convert the barcode dataframe into tube numbers rather than RT primers
    parameters = load_parameters(directory=parameters_path)
    RT_to_sample = pd.read_csv(parameters["RT_to_sample"])
    RT_to_sample.set_index("sample", inplace=True)
    mapping_barcode_table = RT_to_sample["tube"].to_dict()
    for column in barcode_table.columns:
        if column not in RT_to_sample.index.values:
            barcode_table.drop(columns=column, inplace=True)
    barcode_table.rename(columns=mapping_barcode_table, inplace=True)
    # drop the non-existant tubes, added so that there wasn't gaps in RT to sample
    if 0 in barcode_table.columns:
        barcode_table = barcode_table.drop(0, axis=1)
    for tube_to_group in parameters["rois_to_combine"]:
        if parameters["rois_combined_pre_RT_barcodes"] == False:
            barcode_table[tube_to_group] = barcode_table[
                parameters["rois_to_combine"][tube_to_group]
            ].sum(axis=1)
            drop_list = []
            for tube in parameters["rois_to_combine"][tube_to_group]:
                if tube != tube_to_group:
                    drop_list.append(tube)
            barcode_table.drop(columns=drop_list, inplace=True)
    # now remove any samples that have been excluded in parameters yaml
    if parameters["samples_to_drop"]:
        list_samples = [
            x for x in parameters["samples_to_drop"] if x in barcode_table.columns
        ]  # amend list if they aren't in barcode table columns, as some are aggregated in rois_to_combine
        barcode_table.drop(columns=list_samples, inplace=True)
    return barcode_table


def get_id(id):
    """Function to get acronymn from number id
    Args:
        id(num): id for region name
    Returns:
        acronym(str)
    """
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=True)
    if id > 0:
        if bg_atlas.structures[id]["acronym"][-1].isnumeric():
            id = bg_atlas.structures[id]["structure_id_path"][
                -2
            ]  # moving one level up the hierarchy if cortical layer
        elif (
            bg_atlas.structures[id]["acronym"][-2:] == "6a"
            or bg_atlas.structures[id]["acronym"][-2:] == "6b"
        ):
            id = bg_atlas.structures[id]["structure_id_path"][
                -2
            ]  # moving one level up the hierarchy if layer 6a/6b
        newid = bg_atlas.structures[id]["acronym"]
        group_structures = [
            "HY",
            "CB",
            "MY",
            "P",
            "fiber tracts",
            "STR",
            "IPN",
            "BLA",
            "PAL",
            "HPF",
            "SCm",
            "SCs",
            "IC",
            "LGd",
            "LGv",
            "root",
            "SSp",
            "MOB",
            "AOB",
        ]
        olfactory_bulb = ["MOB", "AOB"]
        try:
            up_6 = bg_atlas.structures[id]["structure_id_path"][-6]
            if bg_atlas.structures[up_6]["acronym"] in group_structures:
                newid = bg_atlas.structures[up_6]["acronym"]
        except IndexError:
            print(f'cannot go five higher for {bg_atlas.structures[id]["acronym"]}')
        try:
            up_5 = bg_atlas.structures[id]["structure_id_path"][-5]
            if bg_atlas.structures[up_5]["acronym"] in group_structures:
                newid = bg_atlas.structures[up_5]["acronym"]
        except IndexError:
            print(f'cannot go four higher for {bg_atlas.structures[id]["acronym"]}')
        try:
            up_4 = bg_atlas.structures[id]["structure_id_path"][-4]
            if bg_atlas.structures[up_4]["acronym"] in group_structures:
                newid = bg_atlas.structures[up_4]["acronym"]
        except IndexError:
            print(f'cannot go three higher for {bg_atlas.structures[id]["acronym"]}')
        try:
            up_3 = bg_atlas.structures[id]["structure_id_path"][-3]
            if bg_atlas.structures[up_3]["acronym"] in group_structures:
                newid = bg_atlas.structures[up_3]["acronym"]
        except IndexError:
            print(f'cannot go two higher for {bg_atlas.structures[id]["acronym"]}')
        try:
            up_2 = bg_atlas.structures[id]["structure_id_path"][-2]
            if bg_atlas.structures[up_2]["acronym"] in group_structures:
                newid = bg_atlas.structures[up_2]["acronym"]
        except IndexError:
            print(f'cannot go one higher for {bg_atlas.structures[id]["acronym"]}')

        if newid in olfactory_bulb:
            newid = "OB"
        # ancestors= bg_atlas.get_structure_ancestors(id)
        # newid = [item for item in group_structures if item in ancestors]
        return newid


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="50G", partition="ncpu"),
)
def get_main_region(sample_vol, parameters_path):
    """
    Function to add to table what the main region is and relative proportions
    Args:
        sample_vol: path to pickle file from get generate_region_table_across_samples output
        parameters_path: path to folder containing parameters
    Return:
        None
    """
    # first let's define the A1 source sites by taking the barcodes that have max with min amount A1
    parameters = load_parameters(directory=parameters_path)
    sample_vol_and_regions_table = pd.read_pickle(sample_vol)
    sample_vol_and_regions_table["regions"] = "NA"
    sample_vol_and_regions_table["breakdown"] = "NA"
    # sample_vol_and_regions_table['vol_in_atlas'] = 0
    sample_vol_and_regions_table["main"] = "NA"
    sample_vol_and_regions_table["main_fraction"] = 0

    for index, row in sample_vol_and_regions_table.iterrows():
        all_regions = sample_vol_and_regions_table.loc[index]["Brain Regions"]
        # all_regions = [i for i in all_regions if check_non_target(i, [997, 1009])] #997 are 'fibre tracts' and 1009 is 'root', we don't want to include these in the analysis
        all_reg_converted = []
        all_reg, counts = np.unique(all_regions, return_counts=True)
        for i in all_reg:
            converted = fpf.get_id(i)
            all_reg_converted.append(converted)
        if row["ROI Number"] in parameters["contra"]:
            all_reg_converted = ["Contra-" + s for s in all_reg_converted]
        region_counts = pd.DataFrame({"Regions": all_reg_converted, "Counts": counts})
        sum_values = region_counts.groupby("Regions").sum()
        sum_values = sum_values.sort_values(
            by="Counts", ascending=False, ignore_index=False
        )
        sum_values["Fraction"] = sum_values["Counts"] / sum_values["Counts"].sum()
        sample_vol_and_regions_table.loc[index, "regions"] = str(
            sum_values.index.to_list()
        )
        sample_vol_and_regions_table.loc[index, "breakdown"] = str(
            sum_values.Fraction.to_list()
        )
        sample_vol_and_regions_table.loc[index, "vol_in_atlas"] = sum_values[
            "Counts"
        ].sum()
        sample_vol_and_regions_table.loc[index, "main"] = sum_values.iloc[0].name
        sample_vol_and_regions_table.loc[index, "main_fraction"] = sum_values.iloc[
            0
        ].Fraction
    sample_vol_and_regions_table.to_pickle(sample_vol)


def load_parameters(directory):
    """Load the parameters yaml file containing all the parameters required for
    preprocessing MAPseq data

    Args:
    directory (str): Directory where to load parameters from. Default 'root' for the
        default parameters (found in `mapseq_preprocessing/parameters.py`).

    Returns:
        dict: contents of parameters.yml
    """

    def flatten_dict(d):
        flattened_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened_dict[subkey] = subvalue
            else:
                flattened_dict[key] = value
        return flattened_dict

    if directory == "root":
        parameters_file = Path(__file__).parent / "parameters.yml"
    else:
        parameters_file = Path(directory) / "parameters.yml"
    with open(parameters_file, "r") as f:
        parameters = flatten_dict(yaml.safe_load(f))
    return parameters


def homog_across_cubelet(
    parameters_path,
    cortical,
    shuffled,
    barcode_matrix,
    CT_PT_only=False,
    IT_only=False,
    area_threshold=0.1,
    binary=False, remove_AUDp_vis_cub=False,
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
    parameters = load_parameters(directory=parameters_path)

    # barcodes_across_sample = pd.read_pickle(sequencing_directory / "A1_barcodes.pkl")
    barcodes_across_sample = barcode_matrix.copy()

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    lcm_directory = processed_path / (
        "turnerb_" + parameters["lcm_directory"].split("turnerb_")[1]
    )
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
    ].tolist()  # added as i've increased the threshold for testing
    areas_only_grouped.drop(zero_cols, inplace=True)
    barcodes_across_sample.drop(columns=zero_cols, inplace=True)
    areas_matrix = areas_only_grouped.to_numpy()
    total_frac = np.sum(areas_matrix, axis=1)
    frac_matrix = areas_matrix / total_frac[:, np.newaxis]
    weighted_frac_matrix = frac_matrix / areas_matrix.sum(axis=0)
    # barcodes_sum = barcodes_across_sample.sum(axis=1)
    # barcodes_across_sample =barcodes_across_sample.div(barcodes_sum, axis=0)
    barcodes = barcodes_across_sample.to_numpy()
    if shuffled and not binary:
        barcodes = send_to_shuffle(barcodes=barcodes)
    if binary and shuffled:
        barcodes = send_for_curveball_shuff(barcodes=barcodes)
    #total_projection_strength = np.sum(barcodes, axis=1)  # changed as normalised before
    #barcodes = barcodes / total_projection_strength[:, np.newaxis] changed as normalised to density instead
    bc_matrix = np.matmul(barcodes, weighted_frac_matrix)
    bc_matrix = pd.DataFrame(
        data=bc_matrix,
        columns=areas_only_grouped.columns.to_list(),
        index=barcodes_across_sample.index,
    )
    # bool_barcodes= barcodes_across_sample.astype(bool).astype(int).to_numpy()
    # vol_matrix = np.matmul(bool_barcodes, areas_matrix)
    # bc_matrix = bc_matrix/vol_matrix
    bc_matrix = bc_matrix.dropna(axis=1, how="all")
    bc_matrix = bc_matrix.loc[~(bc_matrix == 0).all(axis=1)]
    row_min    = bc_matrix.min(axis=1)                         
    row_range  = bc_matrix.max(axis=1) - row_min               
    row_range.replace(0, np.nan, inplace=True)                

    bc_matrix  = bc_matrix.sub(row_min,   axis=0)             
    bc_matrix  = bc_matrix.div(row_range, axis=0)    #subtract min and divide by range so max = 1          
    #bc_matrix  = bc_matrix.fillna(0)  
    # bc_matrix =bc_matrix.reset_index(drop=True)
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


def homog_across_area(
    parameters_path, barcode_matrix, cortical, shuffled, IT_only=False, binary=False
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
    parameters = load_parameters(directory=parameters_path)
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
    # barcodes_across_sample = pd.read_pickle(sequencing_directory / "A1_barcodes.pkl")
    barcodes_across_sample = barcode_matrix
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    lcm_directory = processed_path / (
        "turnerb_" + parameters["lcm_directory"].split("turnerb_")[1]
    )
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
        barcodes_homog[barcodes_homog < 0] = (
            0  # this is not ideal, but using sparse logistic regression, there isn't an option to contrain coef to be non-negative
        )
        row_min    = barcodes_homog.min(axis=1)                         
        row_range  = barcodes_homog.max(axis=1) - row_min               
        row_range.replace(0, np.nan, inplace=True)                

        barcodes_homog  = barcodes_homog.sub(row_min,   axis=0)             
        barcodes_homog  = barcodes_homog.div(row_range, axis=0)    #subtract min and divide by range so max = 1          
        barcodes_homog  = barcodes_homog.fillna(0)  
    return barcodes_homog


def get_visual_areas(matrix):
    """
    Function to just look at visual areas in barcode matrix
    """
    vis_areas_matrix = matrix.filter(like="VIS")
    vis_areas_matrix = vis_areas_matrix.drop(columns="VISC")
    # drop zeros
    vis_areas_matrix = vis_areas_matrix.loc[~(vis_areas_matrix == 0).all(axis=1)]
    return vis_areas_matrix


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="12:00:00", mem="8G", partition="ncpu"),
)
def generate_shuffle_population(
    mice, proj_folder, total_number_shuffles, mice_sep=False
):
    """
    Function to generate a population of random barcode shuffles within lcm based on the mice you provide
    Args:
        mice(list): list of mice you want to analyse
        proj_folder: path to folder where the mice datasets are (e.g. "/camp/lab/znamenskiyp/home/shared/projects/turnerb_A1_MAPseq")
    """
    num_shuf_chunk = 200
    number_jobs = int(total_number_shuffles / num_shuf_chunk)
    job_ids = []
    temp_shuffle_folder = Path(proj_folder) / "temp_shuffles"
    temp_shuffle_folder.mkdir(parents=True, exist_ok=True)
    combined_dict_cubelet = {}
    for mouse in mice:
        parameters_path = f"/camp/lab/znamenskiyp/home/shared/projects/turnerb_A1_MAPseq/{mouse}/Sequencing"
        barcodes = pd.read_pickle(f"{parameters_path}/A1_barcodes_thresholded.pkl")
        combined_dict_cubelet[mouse] = homog_across_cubelet(
            parameters_path=parameters_path,
            barcode_matrix=barcodes,
            cortical=True,
            shuffled=False,
            binary=True,
            IT_only=True,
        )
    if len(mice) > 1:
        common_columns_cubelet = set(combined_dict_cubelet[mice[0]].columns)
        for mouse in mice[1:]:
            common_columns_cubelet = common_columns_cubelet.intersection(
                combined_dict_cubelet[mouse].columns
            )

        common_columns_cubelet = list(common_columns_cubelet)
    elif len(mice) == 1:
        common_columns_cubelet = combined_dict_cubelet[mouse].columns
    for new_job in range(number_jobs):
        kwargs = {
                "mice": mice,
                "temp_shuffle_folder": str(temp_shuffle_folder),
                "iteration": new_job,
                "proj_folder": proj_folder,
                "cubelet_cols": common_columns_cubelet,
                "use_slurm": True,
                "slurm_folder": "/camp/home/turnerb/slurm_logs",
                "scripts_name": f"get_shuffled_pop{'_sep' if mice_sep else ''}_{new_job}"
            }
        if mice_sep:
            kwargs["num_shuffles"] = num_shuf_chunk
            job_id = get_shuffles_mice_sep(**kwargs)
        else:
            kwargs["num_chunk"] = num_shuf_chunk
            job_id = get_shuffles(**kwargs)
        job_ids.append(job_id)
    job_ids_adj = ":".join(map(str, job_ids))
    job=collate_all_shuffles(
            temp_shuffle_folder=str(temp_shuffle_folder),
            use_slurm=True,
            slurm_folder="/camp/home/turnerb/slurm_logs",
            job_dependency=job_ids_adj,
            mice=mice,
            mice_sep=mice_sep,
            scripts_name="collating_shuffled_pop",
        )
    print(f'sent collating job {job}')
    #run check to re-run jobs if any failed
    # check_and_run_missing_scripts(prefix="shuffled__neuron_numbers_cubelet", total= number_jobs, mice = mice, temp_shuffle_folder=str(temp_shuffle_folder), mice_sep=mice_sep, use_slurm=True,
    #     slurm_folder="/camp/home/turnerb/slurm_logs", job_dependency=job_ids_adj,)
    

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="1:00:00", mem="1G", partition="ncpu"),
)
def check_and_run_missing_scripts(total, mice, temp_shuffle_folder, mice_sep, prefix= 'shuffled__neuron_numbers_cubelet', path_to_jobs='/camp/home/turnerb/slurm_logs'):
    """Function to check that all the scripts have run for shuffle population, and to run the missing jobs"""
    missing = []
    job_ids = []
    script_name = "get_shuffled_pop_"
    if mice_sep:
        script_name = f"{script_name}sep_"
    for i in range(total):
        filename = f"{prefix}_{i}.pkl"
        if mice_sep:
            mouse_to_look = mice[0]
            filename = f"{prefix}_{mouse_to_look}_{i}.pkl"
        filepath = os.path.join(temp_shuffle_folder, filename)
        if not os.path.isfile(filepath):
            missing.append(i)

    if not missing:
        print("All files are present.")
        collate_all_shuffles(
            temp_shuffle_folder=str(temp_shuffle_folder),
            use_slurm=True,
            slurm_folder=path_to_jobs,
            mice=mice,
            mice_sep=mice_sep,
            scripts_name="collating_shuffled_pop",
        )
    else:
        print(f"Missing files: {missing}")
        for number in missing:
            new_script_name = f"{path_to_jobs}/{script_name}{number}.sh"
            try:
                print(f"Running script: {script_name}")
                result = subprocess.run(
                    ["sbatch", new_script_name],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # Parse SLURM job ID from output
                output = result.stdout.strip()
                if "Submitted batch job" in output:
                    job_id = output.split()[-1]
                    job_ids.append(job_id)
                    print(f"Submitted {new_script_name} as job {job_id}")
                else:
                    print(f"Unexpected sbatch output: {output}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit {new_script_name}: {e.stderr.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to run {new_script_name}: {e}")
        job_ids_adj = ":".join(map(str, job_ids))
        collate_all_shuffles(
            temp_shuffle_folder=str(temp_shuffle_folder),
            use_slurm=True,
            slurm_folder=path_to_jobs,
            job_dependency=job_ids_adj,
            mice=mice,
            mice_sep=mice_sep,
            scripts_name="collating_shuffled_pop",
        )


def create_intermediate_jobs(job_ids, chunk_size=5000):
    """
    Create intermediate jobs if the number of job IDs is greater than 10,000.
    Each intermediate job is dependent on a chunk of the original list.

    Args:
        job_ids (list): List of original job IDs.
        chunk_size (int): Number of job IDs per chunk for intermediate jobs.

    Returns:
        list: List of intermediate job IDs.
    """
    # if  total number of job IDs is less than or equal to 10,000, return the original list
    if len(job_ids) <= 10000:
        return job_ids

    # split job IDs into chunks of `chunk_size`
    def split_into_chunks(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    job_id_chunks = list(split_into_chunks(job_ids, chunk_size))
    # generate an intermediate job for each chunk of jobs
    intermediate_jobs = []
    for i, chunk in enumerate(job_id_chunks):
        update_chunk = ",".join(map(str, chunk))
        int_id = fpf.intermediate_job(
            number=i,
            use_slurm=True,
            slurm_folder="/camp/home/turnerb/slurm_logs",
            job_dependency=update_chunk,
            scripts_name=f"intermediate_job_{i}",
        )
        intermediate_jobs.append(int_id)
    return intermediate_jobs


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="1:00:00", mem="1G", partition="ncpu"),
)
def intermediate_job(number):
    """
    Function to run an intermedidate job with dependency on job chunks
    """
    print(f"chunk set {number} completed")


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=3, time="18:00:00", mem="15G", partition="ncpu"),
)
def get_shuffles(
    mice, temp_shuffle_folder, iteration, proj_folder, cubelet_cols, num_chunk=200
):
    """
    Function to provide a list of 1000 shuffles of your datasets.
    Args:
        mice : list of mice
    """
    # first let's get area projections for n shuffle replicates
    num_shuffles = num_chunk
    warnings.filterwarnings("ignore")
    combined_dict_cubelet = get_shuffled_mouse_populations(
        mice=mice, proj_folder=proj_folder, num_shuffles=num_shuffles
    )
    common_columns_cubelet = cubelet_cols
    column_combinations = list(itertools.combinations(common_columns_cubelet, 2))
    tot_neuron_num_cubelet = []
    probability_data = []
    conditional_prob_data = []
    neuron_numbers_data = []
    cosine_sim_binary_data = []
    for i in range(num_shuffles):
        if len(mice) > 1:
            matrix = pd.concat(
                [
                    combined_dict_cubelet[mouse][i][common_columns_cubelet]
                    for mouse in mice
                ],
                ignore_index=True,
            )
        elif len(mice) == 1:
            matrix = combined_dict_cubelet[mouse][i][common_columns_cubelet]
        tot_neurons = len(matrix)
        neuron_counts = matrix.astype(bool).sum(axis=0).to_dict()
        dict_to_add = {}
        cosine_dict_binary = {}
        cond_prob_dict = {}

        for col_a, col_b in column_combinations:
            # calculate co-projection and correlations
            co_projection = (
                matrix[col_a].astype(bool) & matrix[col_b].astype(bool)
            ).sum()
            dict_to_add[f"{col_a}, {col_b}"] = co_projection
    
            # calculate conditional probabilities
            col_a_project = matrix[matrix[col_a] > 0].astype(bool)
            col_b_project = matrix[matrix[col_b] > 0].astype(bool)
            cond_prob_dict[f"{col_a}, {col_b}"] = (
                col_a_project[col_b].mean() if not col_a_project.empty else np.nan
            )  # for some in shuffle, no longer any projections to certain areas using the homog across area approach, here we put nan in place (likely un-needed with homog across cubelet)
            cond_prob_dict[f"{col_b}, {col_a}"] = (
                col_b_project[col_a].mean() if not col_b_project.empty else np.nan
            )

            # calculate cosine similarities in mean projections (normal and binarized (aka conditional prob))
            for which_comp_type in ["binary"]:  # ["norm", "binary"]:
                if which_comp_type == "binary":
                    neurons_1_av = matrix[matrix[col_a] > 0].astype(bool).mean(axis=0)
                    neurons_2_av = matrix[matrix[col_b] > 0].astype(bool).mean(axis=0)
                else:
                    neurons_1_av = matrix[matrix[col_a] > 0].mean(axis=0)
                    neurons_2_av = matrix[matrix[col_b] > 0].mean(axis=0)

                if (
                    not neurons_1_av.drop([col_a, col_b]).empty
                    and not neurons_2_av.drop([col_a, col_b]).empty
                ):
                    neurons_1_av_arr = (
                        neurons_1_av.drop([col_a, col_b]).to_numpy().reshape(1, -1)
                    )  # drop the columns that are conditioned on
                    neurons_2_av_arr = (
                        neurons_2_av.drop([col_a, col_b]).to_numpy().reshape(1, -1)
                    )
                    cosine_sim = cosine_similarity(neurons_1_av_arr, neurons_2_av_arr)[
                        0
                    ][0]
                    if which_comp_type == "binary":
                        cosine_dict_binary[f"{col_a}, {col_b}"] = cosine_sim
                    else:
                        cosine_dict[f"{col_a}, {col_b}"] = cosine_sim
                else:
                    if which_comp_type == "binary":
                        cosine_dict_binary[f"{col_a}, {col_b}"] = np.nan
                    else:
                        cosine_dict[f"{col_a}, {col_b}"] = np.nan

        # collect data
        tot_neuron_num_cubelet.append(tot_neurons)
        probability_data.append(dict_to_add)
        neuron_numbers_data.append(neuron_counts)
        cosine_sim_binary_data.append(cosine_dict_binary)
        conditional_prob_data.append(cond_prob_dict)

    # final concatenation outside loop
    probability_cubelet = pd.DataFrame(probability_data)
    neuron_numbers_cubelet = pd.DataFrame(neuron_numbers_data)
    cosine_sim_matrix_cubelet_binary = pd.DataFrame(cosine_sim_binary_data)
    conditional_prob_cubelet = pd.DataFrame(conditional_prob_data)
    neuron_num_pandas = pd.DataFrame(tot_neuron_num_cubelet)
    cosine_sim_matrix_cubelet_binary.to_pickle(
        f"{temp_shuffle_folder}/shuffled_cubelet_cosine_sim_binary_{iteration}.pkl"
    )
    probability_cubelet.to_pickle(
        f"{temp_shuffle_folder}/shuffled_cubelet_2_comb_{iteration}.pkl"
    )
    conditional_prob_cubelet.to_pickle(
        f"{temp_shuffle_folder}/shuffled_cubelet_conditional_prob_{iteration}.pkl"
    )
    neuron_numbers_cubelet.to_pickle(
        f"{temp_shuffle_folder}/shuffled__neuron_numbers_cubelet_{iteration}.pkl"
    )
    neuron_num_pandas.to_pickle(
        f"{temp_shuffle_folder}/total_neuron_numbers_cubelet_{iteration}.pkl"
    )


def process_shuffles(mouse, proj_folder, num_shuffles):
    """Helper function to process shuffles for a single mouse"""
    homog_across_cubelet_dict = {}
    parameters_path = f"{proj_folder}/{mouse}/Sequencing"
    parameters = load_parameters(directory=parameters_path)
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
    barcodes_across_sample = pd.read_pickle(
        sequencing_directory / "A1_barcodes_thresholded.pkl"
    )
    print(f"finished generating area matrix for {mouse}")

    for i in range(num_shuffles):
        homog_across_cubelet_dict[i] = homog_across_cubelet(
            parameters_path=parameters_path,
            barcode_matrix=barcodes_across_sample,
            cortical=True,
            shuffled=True,
            binary=True,
            IT_only=True,
        )
    return mouse, homog_across_cubelet_dict


def get_shuffled_mouse_populations(mice, proj_folder, num_shuffles):
    """
    Function to get shuffles of each dataframe for each mouse
    Returns:
    dictionaries of shuffled dataframes
    """
    warnings.filterwarnings("ignore")
    combined_dict_cubelet = {}

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        results = executor.map(
            process_shuffles,
            mice,
            [proj_folder] * len(mice),
            [num_shuffles] * len(mice),
        )

        for mouse, homog_across_cubelet_dict in results:
            combined_dict_cubelet[mouse] = homog_across_cubelet_dict

    return combined_dict_cubelet

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="12:00:00", mem="16G", partition="ncpu"),
)
def get_shuffles_mice_sep(
    mice, temp_shuffle_folder, iteration, proj_folder, cubelet_cols, num_shuffles
):
    """
    Function to provide a list of 1000 shuffles of your datasets. Different to get_shuffles function in that we don't concat the mice together
    Args:
        mice : list of mice
    """
    warnings.filterwarnings("ignore")
    mouse_cubelet_dict = get_shuffled_mouse_populations(
        mice=mice, proj_folder=proj_folder, num_shuffles=num_shuffles
    )
    column_combinations = list(itertools.combinations(cubelet_cols, 2))
    common_columns_cubelet = cubelet_cols
    for mouse in mice:
        tot_neuron_num_cubelet = []
        probability_data = []
        conditional_prob_data = []
        neuron_numbers_data = []
        cosine_sim_binary_data = []
        for i in range(num_shuffles):
            matrix = mouse_cubelet_dict[mouse][i][common_columns_cubelet]
            tot_neurons = len(matrix)
            neuron_counts = matrix.astype(bool).sum(axis=0).to_dict()
            dict_to_add = {}
            cosine_dict_binary = {}
            cond_prob_dict = {}

            for col_a, col_b in column_combinations:
                # calculate co-projection and correlations
                co_projection = (
                    matrix[col_a].astype(bool) & matrix[col_b].astype(bool)
                ).sum()
                dict_to_add[f"{col_a}, {col_b}"] = co_projection
                col_a_project = matrix[matrix[col_a] > 0].astype(bool)
                col_b_project = matrix[matrix[col_b] > 0].astype(bool)
                cond_prob_dict[f"{col_a}, {col_b}"] = (
                    col_a_project[col_b].mean() if not col_a_project.empty else np.nan
                )  # for some in shuffle, no longer any projections to certain areas using the homog across area approach, here we put nan in place (likely un-needed with homog across cubelet)
                cond_prob_dict[f"{col_b}, {col_a}"] = (
                    col_b_project[col_a].mean() if not col_b_project.empty else np.nan
                )

                # calculate cosine similarities in mean projections (normal and binarized (aka conditional prob))
                for which_comp_type in ['binary']:#["norm", "binary"]:
                    if which_comp_type == "binary":
                        neurons_1_av = (
                            matrix[matrix[col_a] > 0].astype(bool).mean(axis=0)
                        )
                        neurons_2_av = (
                            matrix[matrix[col_b] > 0].astype(bool).mean(axis=0)
                        )
                    else:
                        neurons_1_av = matrix[matrix[col_a] > 0].mean(axis=0)
                        neurons_2_av = matrix[matrix[col_b] > 0].mean(axis=0)

                    if (
                        not neurons_1_av.drop([col_a, col_b]).empty
                        and not neurons_2_av.drop([col_a, col_b]).empty
                    ):
                        neurons_1_av_arr = (
                            neurons_1_av.drop([col_a, col_b]).to_numpy().reshape(1, -1)
                        )  # drop the columns that are conditioned on
                        neurons_2_av_arr = (
                            neurons_2_av.drop([col_a, col_b]).to_numpy().reshape(1, -1)
                        )
                        cosine_sim = cosine_similarity(
                            neurons_1_av_arr, neurons_2_av_arr
                        )[0][0]
                        if which_comp_type == "binary":
                            cosine_dict_binary[f"{col_a}, {col_b}"] = cosine_sim
                        else:
                            cosine_dict[f"{col_a}, {col_b}"] = cosine_sim
                    else:
                        if which_comp_type == "binary":
                            cosine_dict_binary[f"{col_a}, {col_b}"] = np.nan
                        else:
                            cosine_dict[f"{col_a}, {col_b}"] = np.nan

            # collect data
            tot_neuron_num_cubelet.append(tot_neurons)
            probability_data.append(dict_to_add)
            neuron_numbers_data.append(neuron_counts)
            cosine_sim_binary_data.append(cosine_dict_binary)
            conditional_prob_data.append(cond_prob_dict)

        # final concatenation outside loop
        probability_cubelet = pd.DataFrame(probability_data)
        neuron_numbers_cubelet = pd.DataFrame(neuron_numbers_data)
        cosine_sim_matrix_cubelet_binary = pd.DataFrame(cosine_sim_binary_data)
        conditional_prob_cubelet = pd.DataFrame(conditional_prob_data)
        neuron_num_pandas = pd.DataFrame(tot_neuron_num_cubelet)
        cosine_sim_matrix_cubelet_binary.to_pickle(
            f"{temp_shuffle_folder}/shuffled_cubelet_cosine_sim_binary_{mouse}_{iteration}.pkl"
        )
        probability_cubelet.to_pickle(
            f"{temp_shuffle_folder}/shuffled_cubelet_2_comb_{mouse}_{iteration}.pkl"
        )
        conditional_prob_cubelet.to_pickle(
            f"{temp_shuffle_folder}/shuffled_cubelet_conditional_prob_{mouse}_{iteration}.pkl"
        )
        neuron_numbers_cubelet.to_pickle(
            f"{temp_shuffle_folder}/shuffled__neuron_numbers_cubelet_{mouse}_{iteration}.pkl"
        )
        neuron_num_pandas.to_pickle(
            f"{temp_shuffle_folder}/total_neuron_numbers_cubelet_{mouse}_{iteration}.pkl"
        )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="100G", partition="ncpu"),
)
def collate_all_shuffles(temp_shuffle_folder, mice_sep, mice, overwrite=True):
    """
    Function to combine the shuffle population tables
    Args:
        temp_shuffle_folder: path to where the folder with the temp shuffles are
        mice_sep: (bool) are the shuffles combined for the separate mice?
        mice: (list) what are the mouse names?
    """
    file_start_names = [
        "shuffled_cubelet_conditional_prob_",
        "shuffled_cubelet_2_comb_",
        "shuffled__neuron_numbers_cubelet_",
        # "shuffled_corr_cubelet_binary_",
        # "shuffled_corr_cubelet_",
        "shuffled_cubelet_cosine_sim_binary_",
        # "shuffled_cubelet_cosine_sim_",
        "total_neuron_numbers_cubelet_",
    ]
    path_to_look = Path(temp_shuffle_folder)
    new_folder = path_to_look.parent / "collated_shuffles"
    new_folder.mkdir(parents=True, exist_ok=True)
    if mice_sep == True:
        files_to_look = []
        for file in file_start_names:
            for mouse in mice:
                files_to_look.append(file + mouse)
    elif mice_sep == False:
        files_to_look = file_start_names
    for file_start in files_to_look:
        #all_files = path_to_look.glob(f"{file_start}*.pkl")
        all_files = list(path_to_look.glob(f"{file_start}*.pkl"))
        all_tables = []
        for f in all_files:
            all_tables.append(pd.read_pickle(f))
        if not all_tables:
            continue
        all_tables = pd.concat(all_tables)
        # if not all_tables.empty:  # only proceed if there are files to collate
        #     all_tables = pd.concat(all_tables)
        save_path = new_folder / f"{file_start}_collated.pkl"
        
        if not overwrite:
            counter = 1
            while save_path.exists():
                counter += 1
                save_path = new_folder / f"{file_start}_collated_{counter}.pkl"
        all_tables.to_pickle(str(save_path))
        print(f'finished saving {file_start}')
        for f in all_files:
            os.remove(f)
        # all_tables.to_pickle(f"{str(new_folder)}/{file_start}_collated_1.pkl")
        # list_all = os.listdir(path_to_look)
        # for file_path in list_all:
        #     if file_path.startswith(file_start):
        #         os.remove(path_to_look / file_path)

def get_AUDp(df):
    """
    Function to get indexes in the sample_vol dataframe that contain main AUDp
    Args:
        df: pandas dataframe with samples and area volumes
    Returns:
        indexes for the df
    """
    indexes = []
    for i, r in df.iterrows():
        if "AUDp" in r["regions"]:
            for region in r["regions"]:
                if region == "AUDp":
                    indexes.append(i)
                    break
    return indexes


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
        lambda row: row.where(row == row.max(), 0), axis=1
    )
    areas_only_grouped = areas_only_grouped.fillna(0)
    areas_only_grouped = areas_only_grouped.loc[
        :, (areas_only_grouped != 0).any(axis=0)
    ]
    return areas_only_grouped


def area_is_main(
    parameters_path, cortical, shuffled, barcode_matrix, IT_only=False, binary=False
):
    """
    Function to output a matrix of neuron barcode distribution across areas, where we assume that the main area in each cubelet is where the barcode counts belong to
    Args:
        parameters_path
        barcode_matrix = pandas dataframe with barcodes
        cortical (bool): True if you want onkly to look at cortical regions
        shuffled (bool): True if you want to shuffle values in all columns as a negative control
    """
    parameters = load_parameters(directory=parameters_path)

    # barcodes_across_sample = pd.read_pickle(sequencing_directory / "A1_barcodes.pkl")
    barcodes_across_sample = barcode_matrix
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    lcm_directory = processed_path / (
        "turnerb_" + parameters["lcm_directory"].split("turnerb_")[1]
    )
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
    #barcodes = barcodes / total_projection_strength[:, np.newaxis] #we don't normalise this anymore
    bc_matrix = np.matmul(barcodes, weighted_frac_matrix)
    bc_matrix = pd.DataFrame(
        data=bc_matrix,
        columns=areas_only_grouped.columns.to_list(),
        index=barcodes_across_sample.index,
    )
    bc_matrix = bc_matrix.dropna(axis=1, how="all")
    bc_matrix = bc_matrix.loc[:, (bc_matrix != 0).any(axis=0)]
    row_min    = bc_matrix.min(axis=1)                         
    row_range  = bc_matrix.max(axis=1) - row_min               
    row_range.replace(0, np.nan, inplace=True)                

    bc_matrix  = bc_matrix.sub(row_min,   axis=0)             
    bc_matrix  = bc_matrix.div(row_range, axis=0)  
    if binary:
        bc_matrix = bc_matrix.astype(bool).astype(int)
    return bc_matrix.fillna(0)

def compare_to_allen(barcode_table, parameters_path):
    """
    Function to compare barcode dataset to allen anterograde tracing datasets
    Args:
        barcode_table (pd dataframe): barcodes across samples dataframe
        parameters_path: path to where params are
    Returns:
        with allen anterograde tracing summed strengths to compare
    """
    mcc = MouseConnectivityCache()
    #     download_allen = Path(
    #     "/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/Allen_Connectivity"
    # )
    parameters = load_parameters(directory=parameters_path)
    lcm_dir = parameters["lcm_directory"]
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    # replace everything up to turnerb_MAPseq in lcm_dir with processed_path
    lcm_dir = processed_path / ("turnerb_" + lcm_dir.split("turnerb_")[1])
    ROI_3D = np.load(f"{lcm_dir}/ROI_3D_25.npy")
    # allen anterograde tracing datasets with more than 75% injection site AUDp
    experiment_id_a = 120491896  # AUDp
    experiment_id_b = 116903230  # AUDp, AUDpo, AUDd, AUDv
    experiment_id_c = 100149109  # AUDp and AUDd
    # injection volumes to normalise to (mm3)
    expt_a_inj_vol = 0.097
    expt_b_inj_vol = 0.114
    expt_c_inj_vol = 0.073
    # get projection density for each anterograde tracing expt: values are sum of projecting pixels per voxel.
    expt_a, pd_a_info = mcc.get_projection_density(experiment_id_a)
    expt_b, pd_b_info = mcc.get_projection_density(experiment_id_b)
    expt_c, pd_c_info = mcc.get_projection_density(experiment_id_c)
    # create an average of three experiments normalised by injection volume
    expt_a_normalised = expt_a / expt_a_inj_vol
    expt_b_normalised = expt_b / expt_b_inj_vol
    expt_c_normalised = expt_c / expt_c_inj_vol
    allen_comp_table = pd.DataFrame(
        columns=[
            "Sample",
            "Allen_expt_a",
            "Allen_expt_b",
            "Allen_expt_c",
            "Mean_Allen",
            "MAPseq_counts",
        ]
    )
    for tube in barcode_table.columns:
        projection_strengths_a = expt_a_normalised[ROI_3D == tube].sum()
        projection_strengths_b = expt_b_normalised[ROI_3D == tube].sum()
        projection_strengths_c = expt_c_normalised[ROI_3D == tube].sum()
        row_data = {
            "Sample": tube,
            "Allen_expt_a": projection_strengths_a,
            "Allen_expt_b": projection_strengths_b,
            "Allen_expt_c": projection_strengths_c,
            "Mean_Allen": np.mean(
                [projection_strengths_a, projection_strengths_b, projection_strengths_c]
            ),
            "MAPseq_counts": barcode_table[tube].sum(),
        }
        allen_comp_table = allen_comp_table.append(row_data, ignore_index=True)
    return allen_comp_table


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
    # np.fill_diagonal(cosine_sim_matrix.values, np.nan)
    return cosine_sim_matrix

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

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

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

