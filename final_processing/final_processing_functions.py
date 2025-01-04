from bg_atlasapi import BrainGlobeAtlas

# from preprocessing_sequencing import preprocess_sequences as ps
from znamutils import slurm_it
import pandas as pd
from final_processing import final_processing_functions as fpf
import numpy as np
import nrrd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import pathlib
import ast
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import itertools
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import os
from sklearn.metrics.pairwise import cosine_similarity
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import yaml
from random import sample, randint, shuffle


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
        parameters_file = pathlib.Path(__file__).parent / "parameters.yml"
    else:
        parameters_file = pathlib.Path(directory) / "parameters.yml"
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
    area_threshold=0.1, binary =False,
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
    sequencing_directory = pathlib.Path(
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
    barcodes_across_sample = barcode_matrix.copy()

    lcm_directory = parameters["lcm_directory"]
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
            "".join([lcm_directory, "/sample_vol_and_regions.pkl"])
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
        barcodes = send_for_curveball_shuff(barcodes = barcodes) 
    total_projection_strength = np.sum(barcodes, axis=1)  # changed as normalised before
    barcodes = barcodes / total_projection_strength[:, np.newaxis]
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
    iter= 5*len(barcodes) #number of iterations of pairs of rows shuffling
    return curve_ball(barcodes, r_presences, num_iterations=iter)

def transfer_ones(matrix, from_column, to_column, percentage):
    # Find indices of ones in column x
    ones_indices = np.where(matrix[:, from_column] == 1)[0]
    # Shuffle the indices
    np.random.shuffle(ones_indices)
    # Calculate the number of ones to transfer
    num_ones_to_transfer = int(percentage * len(ones_indices))
    # Take the first num_ones_to_transfer indices
    transfer_indices = ones_indices[:num_ones_to_transfer]
    # Set ones in column y at the selected indices
    matrix[transfer_indices, to_column] = 1
    return matrix


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
        "".join([lcm_directory, "/sample_vol_and_regions.pkl"])
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
    parameters_path, barcode_matrix, cortical, shuffled, IT_only=False, binary = False
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
    sequencing_directory = pathlib.Path(
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
    lcm_directory = parameters["lcm_directory"]
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
            "".join([lcm_directory, "/sample_vol_and_regions.pkl"])
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
        barcodes_matrix = send_for_curveball_shuff(barcodes = barcodes_matrix)
    total_projection_strength = np.sum(barcodes_matrix, axis=1)
    normalised_bc_matrix = barcodes_matrix / total_projection_strength[:, np.newaxis]
    normalised_bc_matrix = normalised_bc_matrix[
        total_projection_strength > 0, :
    ]  # needed as already removed barcodes with no projections but there are otherwise some nan values resulting from no projections in some barcodes after shuffling
    if not binary:
        mdl = Lasso(fit_intercept=False, positive=True)
        mdl.fit(areas_matrix, normalised_bc_matrix.T)
        barcodes_homog = pd.DataFrame(mdl.coef_, columns=areas_only_grouped.columns)
    if binary: #if data is binarized, we will perform logistic regression rather than linear regression
        binarised_bc_matrix = (normalised_bc_matrix > 0).astype(int)
        #mdl = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=1000)
        mdl = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False, max_iter=1000, C=1.0)
        coef_list = []
        for i in range(binarised_bc_matrix.shape[0]):
            mdl.fit(areas_matrix, binarised_bc_matrix[i, :])  # Fit logistic regression to each row (barcode)
            coef_list.append(mdl.coef_[0])  # Store the coefficients for this barcode
        barcodes_homog = pd.DataFrame(coef_list, columns=areas_only_grouped.columns)
        barcodes_homog[barcodes_homog < 0] = 0 #this is not ideal - using sparse logistic regression, there isn't an option to contrain coef to be non-negative
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
    num_shuf_chunk = 3000
    number_jobs = int(total_number_shuffles / num_shuf_chunk)
    job_ids = []
    temp_shuffle_folder = pathlib.Path(proj_folder) / "temp_shuffles"
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
    if mice_sep == False:
        for new_job in range(number_jobs):
            job_id = get_shuffles(
                mice=mice,
                temp_shuffle_folder=str(temp_shuffle_folder),
                iteration=new_job,
                proj_folder=proj_folder,
                cubelet_cols=common_columns_cubelet, num_chunk = num_shuf_chunk,
                use_slurm=True,
                slurm_folder="/camp/home/turnerb/slurm_logs",
                scripts_name=f"get_shuffled_pop_{new_job}",
            )
            job_ids.append(job_id)
        job_ids = ",".join(map(str, job_ids))
        job_ids_adj = create_intermediate_jobs(job_ids)
        job = collate_all_shuffles(
            temp_shuffle_folder=str(temp_shuffle_folder),
            use_slurm=True,
            slurm_folder="/camp/home/turnerb/slurm_logs",
            job_dependency=job_ids_adj,
            mice=mice,
            mice_sep=False,
        )
    elif mice_sep == True:
        for new_job in range(number_jobs):
            job_id = get_shuffles_mice_sep(
                mice=mice,
                temp_shuffle_folder=str(temp_shuffle_folder),
                iteration=new_job,
                proj_folder=proj_folder,
                cubelet_cols=common_columns_cubelet,
                use_slurm=True, num_shuffles = num_shuf_chunk,
                slurm_folder="/camp/home/turnerb/slurm_logs",
                scripts_name=f"get_shuffled_pop_sep_{new_job}",
            )
            job_ids.append(job_id)
        job_ids = ",".join(map(str, job_ids))
        job_ids_adj = create_intermediate_jobs(job_ids)
        job = collate_all_shuffles(
            temp_shuffle_folder=str(temp_shuffle_folder),
            use_slurm=True,
            slurm_folder="/camp/home/turnerb/slurm_logs",
            job_dependency=job_ids_adj,
            mice=mice,
            mice_sep=True,
            scripts_name="collating_shuffled_pop_sep_mice",
        )
    print(f"collate_all_shuffles= {job}")

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
            yield lst[i:i + chunk_size]

    job_id_chunks = list(split_into_chunks(job_ids, chunk_size))
    # generate an intermediate job for each chunk of jobs
    intermediate_jobs = []
    for i, chunk in enumerate(job_id_chunks):
        int_id = intermediate_job(number=i,
                slurm_folder="/camp/home/turnerb/slurm_logs", job_dependency=chunk,
                scripts_name=f"intermediate_job_{i}",
            )
        intermediate_jobs.append(int_id)
    new_job_ids = ",".join(map(str, intermediate_jobs))

    return new_job_ids

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="1:00:00", mem="1G", partition="ncpu"),
)
def intermediate_job(number):
    """
    Function to run an intermedidate job with dependency on job chunks
    """
    print(f'chunk set {number} completed')

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="16:00:00", mem="16G", partition="ncpu"),
)
def get_shuffles(
    mice,
    temp_shuffle_folder,
    iteration,
    proj_folder,
    cubelet_cols, num_chunk =3000
):
    """
    Function to provide a list of 1000 shuffles of your datasets.
    Args:
        mice : list of mice
    """
    # first let's get area projections for 1000 shuffle replicates
    num_shuffles = num_chunk
    warnings.filterwarnings("ignore")
    combined_dict_cubelet = get_shuffled_mouse_populations(mice=mice, proj_folder=proj_folder, num_shuffles=num_shuffles)
    common_columns_cubelet = cubelet_cols
    # for col_a, col_b in itertools.combinations(common_columns_cubelet, 2):
    #     combination_to_add = f"{col_a}, {col_b}"
    #     combinations.append(combination_to_add)
    column_combinations = list(itertools.combinations(common_columns_cubelet, 2))
    tot_neuron_num_cubelet = []
    probability_data = []
    conditional_prob_data = []
    neuron_numbers_data = []
    corr_data = []
    binary_corr_data = []
    cosine_sim_data = []
    cosine_sim_binary_data = []
    for i in range(num_shuffles):
        if len(mice) > 1:
            matrix = pd.concat(
                [combined_dict_cubelet[mouse][i][common_columns_cubelet] for mouse in mice],
                ignore_index=True,
            )
        elif len(mice) == 1:
            matrix = combined_dict_cubelet[mouse][i][common_columns_cubelet]
        tot_neurons = len(matrix)
        neuron_counts = matrix.astype(bool).sum(axis=0).to_dict()
        dict_to_add = {}
        spearman_corr_dict = {}
        binary_corr_dict = {}
        cosine_dict = {}
        cosine_dict_binary = {}
        cond_prob_dict = {}

        for col_a, col_b in column_combinations:
            # calculate co-projection and correlations
            co_projection = (matrix[col_a].astype(bool) & matrix[col_b].astype(bool)).sum()
            dict_to_add[f"{col_a}, {col_b}"] = co_projection
            spearman_corr_dict[f"{col_a}, {col_b}"] = matrix[col_a].corr(matrix[col_b], method="spearman")
            binary_corr_dict[f"{col_a}, {col_b}"] = matrix[col_a].astype(bool).corr(matrix[col_b].astype(bool), method="spearman")

            # calculate conditional probabilities
            col_a_project = matrix[matrix[col_a] > 0].astype(bool)
            col_b_project = matrix[matrix[col_b] > 0].astype(bool)
            cond_prob_dict[f"{col_a}, {col_b}"] = col_a_project[col_b].mean() if not col_a_project.empty else np.nan # for some in shuffle, no longer any projections to certain areas using the homog across area approach, here we put nan in place (likely un-needed with homog across cubelet)
            cond_prob_dict[f"{col_b}, {col_a}"] = col_b_project[col_a].mean() if not col_b_project.empty else np.nan

            # calculate cosine similarities in mean projections (normal and binarized (aka conditional prob))
            for which_comp_type in ["norm", "binary"]:
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

                if not neurons_1_av.drop([col_a, col_b]).empty and not neurons_2_av.drop([col_a, col_b]).empty:
                    neurons_1_av_arr = neurons_1_av.drop([col_a, col_b]).to_numpy().reshape(1, -1) #drop the columns that are conditioned on
                    neurons_2_av_arr = neurons_2_av.drop([col_a, col_b]).to_numpy().reshape(1, -1)
                    cosine_sim = cosine_similarity(neurons_1_av_arr, neurons_2_av_arr)[0][0]
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
        corr_data.append(spearman_corr_dict)
        binary_corr_data.append(binary_corr_dict)
        cosine_sim_data.append(cosine_dict)
        cosine_sim_binary_data.append(cosine_dict_binary)
        conditional_prob_data.append(cond_prob_dict)

    # final concatenation outside loop
    probability_cubelet = pd.DataFrame(probability_data)
    neuron_numbers_cubelet = pd.DataFrame(neuron_numbers_data)
    corr_cubelet = pd.DataFrame(corr_data)
    corr_cubelet_binary = pd.DataFrame(binary_corr_data)
    cosine_sim_matrix_cubelet = pd.DataFrame(cosine_sim_data)
    cosine_sim_matrix_cubelet_binary = pd.DataFrame(cosine_sim_binary_data)
    conditional_prob_cubelet = pd.DataFrame(conditional_prob_data)
    neuron_num_pandas = pd.DataFrame(tot_neuron_num_cubelet)   
    cosine_sim_matrix_cubelet_binary.to_pickle(
        f"{temp_shuffle_folder}/shuffled_cubelet_cosine_sim_binary_{iteration}.pkl"
    )
    cosine_sim_matrix_cubelet.to_pickle(
        f"{temp_shuffle_folder}/shuffled_cubelet_cosine_sim_{iteration}.pkl"
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
    corr_cubelet.to_pickle(
        f"{temp_shuffle_folder}/shuffled_corr_cubelet_{iteration}.pkl"
    )
    corr_cubelet_binary.to_pickle(
        f"{temp_shuffle_folder}/shuffled_corr_cubelet_binary_{iteration}.pkl"
    )
    neuron_num_pandas.to_pickle(
            f"{temp_shuffle_folder}/total_neuron_numbers_cubelet_{iteration}.pkl"
        )


def get_shuffled_mouse_populations(mice, proj_folder, num_shuffles=3000):
    """
    Function to get shuffles of each dataframe for each mouse
    Returns:
    dictionaries of shuffled dataframes
    """
    warnings.filterwarnings("ignore")
    combined_dict_cubelet = {}
    for num, mouse in enumerate(mice):
        homog_across_cubelet_dict = {}
        parameters_path = f"{proj_folder}/{mouse}/Sequencing"
        parameters = load_parameters(directory=parameters_path)
        sequencing_directory = pathlib.Path(
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
        combined_dict_cubelet[mouse] = homog_across_cubelet_dict
    return combined_dict_cubelet

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="12:00:00", mem="16G", partition="ncpu"),
)
def get_shuffles_mice_sep(
    mice,
    temp_shuffle_folder,
    iteration,
    proj_folder,
    cubelet_cols, num_shuffles=3000
):
    """
    Function to provide a list of 1000 shuffles of your datasets. Different to get_shuffles function in that we don't concat the mice together
    Args:
        mice : list of mice
    """
    warnings.filterwarnings("ignore")
    mouse_cubelet_dict = get_shuffled_mouse_populations(mice=mice, proj_folder=proj_folder, num_shuffles=num_shuffles)
    column_combinations = list(itertools.combinations(cubelet_cols, 2))
    common_columns_cubelet = cubelet_cols
    for mouse in mice:
        tot_neuron_num_cubelet = []
        probability_data = []
        conditional_prob_data = []
        neuron_numbers_data = []
        corr_data = []
        binary_corr_data = []
        cosine_sim_data = []
        cosine_sim_binary_data = []
        for i in range(num_shuffles):
            matrix = combined_dict_cubelet[mouse][i][common_columns_cubelet]
            tot_neurons = len(matrix)
            neuron_counts = matrix.astype(bool).sum(axis=0).to_dict()
            dict_to_add = {}
            spearman_corr_dict = {}
            binary_corr_dict = {}
            cosine_dict = {}
            cosine_dict_binary = {}
            cond_prob_dict = {}

            for col_a, col_b in column_combinations:
                # calculate co-projection and correlations
                co_projection = (matrix[col_a].astype(bool) & matrix[col_b].astype(bool)).sum()
                dict_to_add[f"{col_a}, {col_b}"] = co_projection
                spearman_corr_dict[f"{col_a}, {col_b}"] = matrix[col_a].corr(matrix[col_b], method="spearman")
                binary_corr_dict[f"{col_a}, {col_b}"] = matrix[col_a].astype(bool).corr(matrix[col_b].astype(bool), method="spearman")

                # calculate conditional probabilities
                col_a_project = matrix[matrix[col_a] > 0].astype(bool)
                col_b_project = matrix[matrix[col_b] > 0].astype(bool)
                cond_prob_dict[f"{col_a}, {col_b}"] = col_a_project[col_b].mean() if not col_a_project.empty else np.nan # for some in shuffle, no longer any projections to certain areas using the homog across area approach, here we put nan in place (likely un-needed with homog across cubelet)
                cond_prob_dict[f"{col_b}, {col_a}"] = col_b_project[col_a].mean() if not col_b_project.empty else np.nan

                # calculate cosine similarities in mean projections (normal and binarized (aka conditional prob))
                for which_comp_type in ["norm", "binary"]:
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

                    if not neurons_1_av.drop([col_a, col_b]).empty and not neurons_2_av.drop([col_a, col_b]).empty:
                        neurons_1_av_arr = neurons_1_av.drop([col_a, col_b]).to_numpy().reshape(1, -1) #drop the columns that are conditioned on
                        neurons_2_av_arr = neurons_2_av.drop([col_a, col_b]).to_numpy().reshape(1, -1)
                        cosine_sim = cosine_similarity(neurons_1_av_arr, neurons_2_av_arr)[0][0]
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
            corr_data.append(spearman_corr_dict)
            binary_corr_data.append(binary_corr_dict)
            cosine_sim_data.append(cosine_dict)
            cosine_sim_binary_data.append(cosine_dict_binary)
            conditional_prob_data.append(cond_prob_dict)

        # final concatenation outside loop
        probability_cubelet = pd.DataFrame(probability_data)
        neuron_numbers_cubelet = pd.DataFrame(neuron_numbers_data)
        corr_cubelet = pd.DataFrame(corr_data)
        corr_cubelet_binary = pd.DataFrame(binary_corr_data)
        cosine_sim_matrix_cubelet = pd.DataFrame(cosine_sim_data)
        cosine_sim_matrix_cubelet_binary = pd.DataFrame(cosine_sim_binary_data)
        conditional_prob_cubelet = pd.DataFrame(conditional_prob_data)
        neuron_num_pandas = pd.DataFrame(tot_neuron_num_cubelet)   
        cosine_sim_matrix_cubelet_binary.to_pickle(
            f"{temp_shuffle_folder}/shuffled_cubelet_cosine_sim_binary_{mouse}_{iteration}.pkl"
        )
        cosine_sim_matrix_cubelet.to_pickle(
            f"{temp_shuffle_folder}/shuffled_cubelet_cosine_sim_{mouse}_{iteration}.pkl"
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
        corr_cubelet.to_pickle(
            f"{temp_shuffle_folder}/shuffled_corr_cubelet_{mouse}_{iteration}.pkl"
        )
        corr_cubelet_binary.to_pickle(
            f"{temp_shuffle_folder}/shuffled_corr_cubelet_binary_{mouse}_{iteration}.pkl"
        )
        neuron_num_pandas.to_pickle(
                f"{temp_shuffle_folder}/total_neuron_numbers_cubelet_{mouse}_{iteration}.pkl"
            )

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="100G", partition="ncpu"),
)
def collate_all_shuffles(temp_shuffle_folder, mice_sep, mice):
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
        "shuffled_corr_cubelet_binary_",
        "shuffled_corr_cubelet_",
        "shuffled_cubelet_cosine_sim_binary_",
        "shuffled_cubelet_cosine_sim_",
        "total_neuron_numbers_cubelet_",
    ]
    path_to_look = pathlib.Path(temp_shuffle_folder)
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
        all_files = path_to_look.glob(f"{file_start}*.pkl")
        all_tables = []
        for f in all_files:
            all_tables.append(pd.read_pickle(f))
        all_tables = pd.concat(all_tables)
        all_tables.to_pickle(f"{str(new_folder)}/{file_start}_collated.pkl")
        list_all = os.listdir(path_to_look)
        for file_path in list_all:
            if file_path.startswith(file_start):
                os.remove(path_to_look / file_path)


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="200G", partition="ncpu"),
)
def get_three_shuffles(mice):
    """
    Function to provide a list of 1000 shuffles of your datasets and count number of 3 combinations in each
    Args:
        mice : list of mice
    """
    # first let's get area projections for 1000 shuffle replicates
    num_shuffles = 1000
    warnings.filterwarnings("ignore")
    combined_dict_area = {}
    combined_dict_cubelet = {}
    for num, mouse in enumerate(mice):
        homog_across_cubelet = {}
        homog_across_area = {}
        parameters_path = f"/camp/lab/znamenskiyp/home/shared/projects/turnerb_A1_MAPseq/{mouse}/Sequencing"
        parameters = load_parameters(directory=parameters_path)
        sequencing_directory = pathlib.Path(
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
        lcm_directory = parameters["lcm_directory"]
        barcodes_across_sample = barcodes_across_sample[
            barcodes_across_sample.astype(bool).sum(axis=1) > 0
        ]
        areas_only_grouped = get_area_volumes(
            barcode_table_cols=barcodes_across_sample.columns,
            lcm_directory=lcm_directory,
        )
        areas_matrix = areas_only_grouped.to_numpy()
        total_frac = np.sum(areas_matrix, axis=1)
        frac_matrix = areas_matrix / total_frac[:, np.newaxis]
        weighted_frac_matrix = frac_matrix / areas_matrix.sum(axis=0)
        barcodes = barcodes_across_sample.to_numpy()

        for i in range(num_shuffles):
            barcodes_shuffled = fpf.send_to_shuffle(barcodes=barcodes)
            total_projection_strength = np.sum(barcodes_shuffled, axis=1)
            # barcodes_shuffled = barcodes_shuffled.astype(int)/ total_projection_strength[:, np.newaxis]
            bc_matrix = np.matmul(barcodes_shuffled, weighted_frac_matrix)
            bc_matrix = pd.DataFrame(
                data=bc_matrix, columns=areas_only_grouped.columns.to_list()
            )
            bc_matrix = bc_matrix.loc[~(bc_matrix == 0).all(axis=1)]
            homog_across_cubelet[i] = bc_matrix.fillna(0)
            normalised_bc_matrix = barcodes_shuffled[
                total_projection_strength > 0, :
            ]  # needed as already removed barcodes with no projections some nan values if shuffled and no projections in some barcodes

            mdl = Lasso(fit_intercept=False, positive=True)
            mdl.fit(areas_matrix, normalised_bc_matrix.T)
            homog_across_area[i] = pd.DataFrame(
                data=mdl.coef_, columns=areas_only_grouped.columns
            )
        combined_dict_cubelet[mouse] = homog_across_cubelet
        combined_dict_area[mouse] = homog_across_area
    cols = [
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
    combinations = []
    for col_a, col_b, col_c in itertools.combinations(cols, 3):
        combination_to_add = f"{col_a}, {col_b}, {col_c}"
        combinations.append(combination_to_add)
    probability_cubelet = pd.DataFrame(columns=combinations)
    probability_area = pd.DataFrame(columns=combinations)
    for i in range(num_shuffles):
        shuffled_combined_cubelet = pd.concat(
            [
                combined_dict_cubelet["FIAA45.6a"][i][cols],
                combined_dict_cubelet["FIAA45.6d"][i][cols],
            ],
            ignore_index=True,
        )
        shuffled_combined_area = pd.concat(
            [
                combined_dict_area["FIAA45.6a"][i][cols],
                combined_dict_area["FIAA45.6d"][i][cols],
            ],
            ignore_index=True,
        )

        for which, matrix in enumerate(
            [shuffled_combined_cubelet, shuffled_combined_area]
        ):
            dict_to_add = {}

            for col_a, col_b, col_c in itertools.combinations(cols, 3):
                prob_df = pd.DataFrame()
                prob_df["a"] = matrix[col_a].astype(bool)
                prob_df["b"] = matrix[col_b].astype(bool)
                prob_df["c"] = matrix[col_c].astype(bool)
                prob_df["matching"] = prob_df.apply(
                    lambda x: 1 if x["a"] and x["b"] and x["c"] else 0, axis=1
                )
                dict_to_add[f"{col_a}, {col_b}, {col_c}"] = prob_df["matching"].sum()

            if which == 0:
                probability_cubelet = pd.concat(
                    [probability_cubelet, pd.DataFrame(dict_to_add, index=[i])]
                )
            if which == 1:
                probability_area = pd.concat(
                    [probability_area, pd.DataFrame(dict_to_add, index=[i])]
                )

    probability_cubelet.to_pickle(
        "/camp/lab/znamenskiyp/home/shared/code/MAPseq_processing/shuffled_cubelet_3_comb.pkl"
    )
    probability_area.to_pickle(
        "/camp/lab/znamenskiyp/home/shared/code/MAPseq_processing/shuffled_area_3_comb.pkl"
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="12:00:00", mem="16G", partition="ncpu"),
)
def get_anterior_posterior_shuffles(mice, temp_shuffle_folder, iteration, proj_folder):
    """
    Function to provide a list of 1000 shuffles of your datasets.
    Args:
        mice : list of mice
    """
    bg_atlas = BrainGlobeAtlas("allen_mouse_10um", check_latest=False)
    AUDp_id = bg_atlas.structures["AUDp"]["id"]
    mcc = MouseConnectivityCache(resolution=25)
    rsp = mcc.get_reference_space()
    AUDp_mask = rsp.make_structure_mask([AUDp_id], direct_only=False)
    indices_AUDp = np.argwhere(AUDp_mask == 1)

    # select anterior and posterior parts of A1
    max_y = np.max(indices_AUDp[:, 0])
    min_y = np.min(indices_AUDp[:, 0])
    AP_midpoint_A1 = ((max_y - min_y) / 2) + min_y
    posterior_neurons = indices_AUDp[indices_AUDp[:, 0] >= AP_midpoint_A1]
    anterior_neurons = indices_AUDp[indices_AUDp[:, 0] < AP_midpoint_A1]
    # now select only the ipsiliateral side of where was injected
    x_midpoint = AUDp_mask.shape[2] // 2
    contra_mask = np.zeros_like(AUDp_mask, dtype=bool)
    contra_mask[:, :, x_midpoint:] = 1
    min_count = 40
    num_shuffles = 50
    warnings.filterwarnings("ignore")
    combined_dict_area_anterior = {}
    combined_dict_cubelet_anterior = {}
    combined_dict_area_posterior = {}
    combined_dict_cubelet_posterior = {}
    for num, mouse in enumerate(mice):
        homog_across_cubelet = {}
        homog_across_area = {}

        parameters_path = f"{proj_folder}/{mouse}/Sequencing"
        parameters = load_parameters(directory=parameters_path)
        sequencing_directory = pathlib.Path(
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
        barcodes = pd.read_pickle(
            sequencing_directory / "A1_barcodes_thresholded_with_source.pkl"
        )
        lcm_directory = parameters["lcm_directory"]

        # split into anterior and posterior
        ROI_3D = np.load(f"{lcm_directory}/ROI_3D_25.npy")
        AP_samples = {}
        AP_source_filtered = {}
        all_AUDp_samples = np.unique(ROI_3D * AUDp_mask * contra_mask)
        all_AUDp_samples = [sample for sample in all_AUDp_samples if sample != 0]

        for i, index in enumerate([anterior_neurons, posterior_neurons]):
            mask = np.zeros_like(AUDp_mask, dtype=bool)
            mask[tuple(zip(*index))] = True
            names = ["anterior_neurons", "posterior_neurons"]
            sample_list = np.unique(ROI_3D * mask * contra_mask)
            sample_list = [sample for sample in sample_list if sample != 0]
            AP_samples[names[i]] = sample_list
        for sample in AP_samples[
            "anterior_neurons"
        ]:  # check if some samples are in both anterior and posterior A1 source lists, and if so remove the one that is less frequent on one side
            if sample in AP_samples["posterior_neurons"]:
                anterior_count = sum(ROI_3D[tuple(zip(*anterior_neurons))] == sample)
                posterior_count = sum(ROI_3D[tuple(zip(*posterior_neurons))] == sample)
                if anterior_count > posterior_count:
                    AP_samples["posterior_neurons"].remove(sample)
                if anterior_count < posterior_count:
                    AP_samples["anterior_neurons"].remove(sample)
        for number, key in enumerate(AP_samples):
            filtered_barcodes_source = barcodes[
                barcodes.idxmax(axis=1).isin(AP_samples[key])
            ]
            source_removed_barcodes = filtered_barcodes_source.drop(
                columns=all_AUDp_samples
            )  # drop the A1 containing regions
            barcodes_across_sample = source_removed_barcodes[
                source_removed_barcodes.sum(axis=1) > min_count
            ]

            # for i, index in enumerate([anterior_neurons, posterior_neurons]):
            #     mask = np.zeros_like(AUDp_mask, dtype=bool)
            #     mask[tuple(zip(*index))] = True
            #     names = ['anterior_neurons', 'posterior_neurons']
            #     sample_list = np.unique(ROI_3D *  mask * contra_mask)
            #     sample_list = [sample for sample in sample_list if sample != 0]
            #     AP_samples[names[i]] = sample_list
            # for sample in AP_samples['anterior_neurons']: #check if some samples are in both anterior and posterior A1 source lists, and if so remove the one that is less frequent on one side
            #     if sample in AP_samples['posterior_neurons']:
            #         anterior_count = sum(ROI_3D[tuple(zip(*anterior_neurons))] == sample)
            #         posterior_count = sum(ROI_3D[tuple(zip(*posterior_neurons))] == sample)
            #         if anterior_count>posterior_count:
            #             AP_samples['posterior_neurons'].remove(sample)
            #         if anterior_count<posterior_count:
            #             AP_samples['anterior_neurons'].remove(sample)

            # for number, key in enumerate(AP_samples):
            #     filtered_barcodes_source = barcodes[barcodes.idxmax(axis=1).isin(AP_samples[key])]
            #     barcodes_across_sample = filtered_barcodes_source.drop(columns = all_AUDp_samples)
            # barcodes_across_sample = barcodes_across_sample[
            #     barcodes_across_sample.astype(bool).sum(axis=1) > 0
            # ]
            areas_only_grouped = get_area_volumes(
                barcode_table_cols=barcodes_across_sample.columns,
                lcm_directory=lcm_directory,
            )
            areas_matrix = areas_only_grouped.to_numpy()
            total_frac = np.sum(areas_matrix, axis=1)
            frac_matrix = areas_matrix / total_frac[:, np.newaxis]
            weighted_frac_matrix = frac_matrix / areas_matrix.sum(axis=0)
            barcodes_nump = barcodes_across_sample.to_numpy()
            print(f"finished generating area matrix for {mouse} {key}", flush=True)
            for i in range(num_shuffles):
                barcodes_shuffled = send_to_shuffle(barcodes=barcodes_nump)
                total_projection_strength = np.sum(barcodes_shuffled, axis=1)
                barcodes_shuffled = (
                    barcodes_shuffled / total_projection_strength[:, np.newaxis]
                )
                bc_matrix = np.matmul(barcodes_shuffled, weighted_frac_matrix)
                bc_matrix = pd.DataFrame(
                    data=bc_matrix, columns=areas_only_grouped.columns.to_list()
                )
                bc_matrix = bc_matrix.loc[~(bc_matrix == 0).all(axis=1)]
                homog_across_cubelet[i] = bc_matrix.fillna(0)
                normalised_bc_matrix = barcodes_shuffled[
                    total_projection_strength > 0, :
                ]  # needed as already removed barcodes with no projections some nan values if shuffled and no projections in some barcodes

                mdl = Lasso(fit_intercept=False, positive=True)
                mdl.fit(areas_matrix, normalised_bc_matrix.T)
                homog_across_area[i] = pd.DataFrame(
                    data=mdl.coef_, columns=areas_only_grouped.columns
                )

            if key == "anterior_neurons":
                combined_dict_cubelet_anterior[mouse] = homog_across_cubelet
                combined_dict_area_anterior[mouse] = homog_across_area
            else:
                combined_dict_cubelet_posterior[mouse] = homog_across_cubelet
                combined_dict_area_posterior[mouse] = homog_across_area
    cols = [
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
        "RSPv",
        "RSPd",
        "IC",
        "SCs",
        "STR",
        "RSPagl",
        "SCm",
        "ACAd",
        "ACAv",
        "SSp",
        "SSs",
        "MOp",
        "MOs",
        "TEa",
        "Contra",
        "MGv",
        "LP",
        "LGd",
        "LGv",
        "AUDd",
        "AUDv",
        "HPF",
        "ECT",
        "PERI",
    ]
    # cols = ['VISli','VISpor', 'VISpl', 'VISl', 'VISp', 'VISal', 'VISam', 'VISpm', 'VISa', 'VISrl']
    # common_columns_cubelet = list(set(combined_dict_cubelet['FIAA45.6a'][0].columns).intersection(combined_dict_cubelet['FIAA45.6d'][0].columns))
    # common_columns_area = list(set(combined_dict_area['FIAA45.6a'][0].columns).intersection(combined_dict_area['FIAA45.6d'][0].columns))
    # all_common_columns = [x for x in common_columns_cubelet if x in common_columns_area] #might want to change this if there is any differences - I don't think there is, but I put just in case
    # common_columns_cubelet = ['VISli','VISpor', 'VISpl', 'VISl', 'VISp', 'VISal', 'VISam', 'VISpm', 'VISa', 'VISrl']
    combinations = []
    for col_a, col_b in itertools.combinations(cols, 2):
        combination_to_add = f"{col_a}, {col_b}"
        combinations.append(combination_to_add)
    probability_cubelet_anterior = pd.DataFrame(columns=combinations)
    probability_area_anterior = pd.DataFrame(columns=combinations)
    neuron_numbers_cubelet_anterior = pd.DataFrame(columns=cols)
    neuron_numbers_area_anterior = pd.DataFrame(columns=cols)
    corr_cubelet_anterior = pd.DataFrame(columns=combinations)
    corr_area_anterior = pd.DataFrame(columns=combinations)

    probability_cubelet_posterior = pd.DataFrame(columns=combinations)
    probability_area_posterior = pd.DataFrame(columns=combinations)
    neuron_numbers_cubelet_posterior = pd.DataFrame(columns=cols)
    neuron_numbers_area_posterior = pd.DataFrame(columns=cols)
    corr_cubelet_posterior = pd.DataFrame(columns=combinations)
    corr_area_posterior = pd.DataFrame(columns=combinations)

    for i in range(num_shuffles):
        shuffled_combined_cubelet_anterior = pd.concat(
            [
                combined_dict_cubelet_anterior["FIAA45.6a"][i][cols],
                combined_dict_cubelet_anterior["FIAA45.6d"][i][cols],
            ],
            ignore_index=True,
        )
        shuffled_combined_area_anterior = pd.concat(
            [
                combined_dict_area_anterior["FIAA45.6a"][i][cols],
                combined_dict_area_anterior["FIAA45.6d"][i][cols],
            ],
            ignore_index=True,
        )
        shuffled_combined_cubelet_posterior = pd.concat(
            [
                combined_dict_cubelet_posterior["FIAA45.6a"][i][cols],
                combined_dict_cubelet_posterior["FIAA45.6d"][i][cols],
            ],
            ignore_index=True,
        )
        shuffled_combined_area_posterior = pd.concat(
            [
                combined_dict_area_posterior["FIAA45.6a"][i][cols],
                combined_dict_area_posterior["FIAA45.6d"][i][cols],
            ],
            ignore_index=True,
        )

        # for i in range(num_shuffles):
        #     shuffled_combined_cubelet = pd.concat([combined_dict_cubelet['FIAA45.6a'][i][cols], combined_dict_cubelet['FIAA45.6d'][i][cols]], ignore_index=True)
        #     shuffled_combined_area = pd.concat([combined_dict_area['FIAA45.6a'][i][cols], combined_dict_area['FIAA45.6d'][i][cols]], ignore_index=True)
        for which, matrix in enumerate(
            [
                shuffled_combined_cubelet_anterior,
                shuffled_combined_area_anterior,
                shuffled_combined_cubelet_posterior,
                shuffled_combined_area_posterior,
            ]
        ):
            dict_to_add = {}
            num_dict = {}
            corr_dict = {}
            for column in cols:
                num_dict[column] = matrix[column].astype(bool).sum()
            for col_a, col_b in itertools.combinations(cols, 2):
                prob_df = pd.DataFrame()
                prob_df["a"] = matrix[col_a].astype(bool)
                prob_df["b"] = matrix[col_b].astype(bool)
                prob_df["matching"] = prob_df.apply(
                    lambda x: 1 if x["a"] and x["b"] else 0, axis=1
                )
                dict_to_add[f"{col_a}, {col_b}"] = prob_df["matching"].sum()
                matrix = matrix[matrix.astype(bool).sum(axis=1) > 0].reset_index(
                    drop=True
                )
                corr_dict[f"{col_a}, {col_b}"] = matrix[col_a].corr(
                    matrix[col_b], method="spearman"
                )

            if which == 0:
                probability_cubelet_anterior = pd.concat(
                    [probability_cubelet_anterior, pd.DataFrame(dict_to_add, index=[i])]
                )
                neuron_numbers_cubelet_anterior = pd.concat(
                    [neuron_numbers_cubelet_anterior, pd.DataFrame(num_dict, index=[i])]
                )
                corr_cubelet_anterior = pd.concat(
                    [corr_cubelet_anterior, pd.DataFrame(corr_dict, index=[i])]
                )
            if which == 1:
                probability_area_anterior = pd.concat(
                    [probability_area_anterior, pd.DataFrame(dict_to_add, index=[i])]
                )
                neuron_numbers_area_anterior = pd.concat(
                    [neuron_numbers_area_anterior, pd.DataFrame(num_dict, index=[i])]
                )
                corr_area_anterior = pd.concat(
                    [corr_area_anterior, pd.DataFrame(corr_dict, index=[i])]
                )
            if which == 2:
                probability_cubelet_posterior = pd.concat(
                    [
                        probability_cubelet_posterior,
                        pd.DataFrame(dict_to_add, index=[i]),
                    ]
                )
                neuron_numbers_cubelet_posterior = pd.concat(
                    [
                        neuron_numbers_cubelet_posterior,
                        pd.DataFrame(num_dict, index=[i]),
                    ]
                )
                corr_cubelet_posterior = pd.concat(
                    [corr_cubelet_posterior, pd.DataFrame(corr_dict, index=[i])]
                )
            if which == 3:
                probability_area_posterior = pd.concat(
                    [probability_area_posterior, pd.DataFrame(dict_to_add, index=[i])]
                )
                neuron_numbers_area_posterior = pd.concat(
                    [neuron_numbers_area_posterior, pd.DataFrame(num_dict, index=[i])]
                )
                corr_area_posterior = pd.concat(
                    [corr_area_posterior, pd.DataFrame(corr_dict, index=[i])]
                )
    temp_shuffle_folder = pathlib.Path(proj_folder) / "temp_shuffles"
    probability_cubelet_anterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_cubelet_2_comb_anterior_{iteration}.pkl"
    )
    probability_area_anterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_area_2_comb_anterior_{iteration}.pkl"
    )
    neuron_numbers_area_anterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_neuron_numbers_area_anterior_{iteration}.pkl"
    )
    neuron_numbers_cubelet_anterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_neuron_numbers_cubelet_anterior_{iteration}.pkl"
    )
    corr_area_anterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_corr_area_anterior_{iteration}.pkl"
    )
    corr_cubelet_anterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_corr_cubelet_anterior_{iteration}.pkl"
    )

    probability_cubelet_posterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_cubelet_2_comb_posterior_{iteration}.pkl"
    )
    probability_area_posterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_area_2_comb_posterior_{iteration}.pkl"
    )
    neuron_numbers_area_posterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_neuron_numbers_area_posterior_{iteration}.pkl"
    )
    neuron_numbers_cubelet_posterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_neuron_numbers_cubelet_posterior_{iteration}.pkl"
    )
    corr_area_posterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_corr_area_posterior_{iteration}.pkl"
    )
    corr_cubelet_posterior.to_pickle(
        f"{temp_shuffle_folder}/shuffled_corr_cubelet_posterior_{iteration}.pkl"
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="100G", partition="ncpu"),
)
def collate_all_shuffles_ant_post(temp_shuffle_folder):
    """
    Function to combine the shuffle population tables
    """
    files_to_look = [
        "shuffled_cubelet_2_comb_anterior_",
        "shuffled_area_2_comb_anterior_",
        "shuffled_neuron_numbers_area_anterior_",
        "shuffled_neuron_numbers_cubelet_anterior_",
        "shuffled_corr_area_anterior_",
        "shuffled_corr_cubelet_anterior_",
        "shuffled_cubelet_2_comb_posterior_",
        "shuffled_area_2_comb_posterior_",
        "shuffled_neuron_numbers_area_posterior_",
        "shuffled_neuron_numbers_cubelet_posterior_",
        "shuffled_corr_area_posterior_",
        "shuffled_corr_cubelet_posterior_",
    ]
    path_to_look = pathlib.Path(temp_shuffle_folder)
    new_folder = path_to_look.parent / "collated_shuffles"
    new_folder.mkdir(parents=True, exist_ok=True)
    for file_start in files_to_look:
        all_files = path_to_look.glob(f"{file_start}*.pkl")
        all_tables = []
        for f in all_files:
            all_tables.append(pd.read_pickle(f))
        all_tables = pd.concat(all_tables)
        all_tables.to_pickle(f"{str(new_folder)}/{file_start}collated.pkl")
        list_all = os.listdir(path_to_look)
        for file_path in list_all:
            if file_path.startswith(file_start):
                os.remove(path_to_look / file_path)


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
        "".join([lcm_directory, "/sample_vol_and_regions.pkl"])
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


def area_is_main(parameters_path, cortical, shuffled, barcode_matrix, IT_only=False, binary=False):
    """
    Function to output a matrix of neuron barcode distribution across areas, where we assume that the main area in each cubelet is where the barcode counts belong to
    Args:
        parameters_path
        barcode_matrix = pandas dataframe with barcodes
        cortical (bool): True if you want onkly to look at cortical regions
        shuffled (bool): True if you want to shuffle values in all columns as a negative control
    """
    parameters = load_parameters(directory=parameters_path)
    sequencing_directory = pathlib.Path(
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

    lcm_directory = parameters["lcm_directory"]
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
            "".join([lcm_directory, "/sample_vol_and_regions.pkl"])
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
        barcodes = send_for_curveball_shuff(barcodes = barcodes) 
    total_projection_strength = np.sum(barcodes, axis=1)  # changed as normalised before
    barcodes = barcodes / total_projection_strength[:, np.newaxis]
    bc_matrix = np.matmul(barcodes, weighted_frac_matrix)
    bc_matrix = pd.DataFrame(
        data=bc_matrix,
        columns=areas_only_grouped.columns.to_list(),
        index=barcodes_across_sample.index,
    )
    bc_matrix = bc_matrix.dropna(axis=1, how="all")
    bc_matrix = bc_matrix.loc[:, (bc_matrix != 0).any(axis=0)]
    if binary:
        bc_matrix= bc_matrix.astype(bool).astype(int)
    return bc_matrix.fillna(0)


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="12:00:00", mem="16G", partition="ncpu"),
)
def get_upper_lower_shuffles(mice, temp_shuffle_folder, iteration, proj_folder):
    """
    Function to provide a list of 1000 shuffles of your datasets.
    Args:
        mice : list of mice
    """
    # first let's get area projections for 1000 shuffle replicates
    num_shuffles = 100
    warnings.filterwarnings("ignore")

    combined_matrices = {}
    upper_lower_dict = pd.read_pickle(f"{proj_folder}/upper_lower_dict.pkl")
    layers = ["upper_layer", "lower_layer"]
    shuffle_dict = {}
    cols_to_look = [
        "VISp",
        "VISpor",
        "VISli",
        "VISal",
        "VISl",
        "VISpl",
        "VISpm",
        "VISrl",
        "VISam",
        "VISa",
        "RSPv",
        "RSPd",
        "STR",
        "RSPagl",
        "ACAd",
        "ACAv",
        "SSp",
        "SSs",
        "MOp",
        "MOs",
        "TEa",
        "Contra",
        "AUDd",
        "AUDv",
        "HPF",
        "ECT",
        "PERI",
    ]
    for i in range(num_shuffles):
        for layer in layers:
            combined_dict_proper = {}
            for num, mouse in enumerate(mice):
                parameters_path = f"{proj_folder}/{mouse}/Sequencing"
                parameters = load_parameters(directory=parameters_path)
                lcm_directory = parameters["lcm_directory"]
                barcodes = upper_lower_dict[mouse][layer]
                new_df = fpf.homog_across_cubelet(
                    parameters_path=parameters_path,
                    barcode_matrix=barcodes,
                    cortical=False,
                    shuffled=True,
                    dummy_data=False,
                    IT_only=True,
                )
                combined_dict_proper[mouse] = new_df
            common_columns = set(
                combined_dict_proper["FIAA45.6a"].columns
            ).intersection(combined_dict_proper["FIAA45.6d"].columns)
            combined_matrices[layer] = pd.concat(
                [
                    combined_dict_proper["FIAA45.6a"][common_columns],
                    combined_dict_proper["FIAA45.6d"][common_columns],
                ],
                ignore_index=False,
            )

        upper_matrix = combined_matrices["upper_layer"][cols_to_look]
        lower_matrix = combined_matrices["lower_layer"][cols_to_look]
        for number, col in enumerate(cols_to_look):
            upper_P = (
                upper_matrix[upper_matrix[col] > 0]
                .astype(bool)
                .astype(int)
                .mean(axis=0)
            )
            upper_odds = upper_P / (1 - upper_P)
            lower_P = (
                lower_matrix[lower_matrix[col] > 0]
                .astype(bool)
                .astype(int)
                .mean(axis=0)
            )
            lower_odds = lower_P / (1 - lower_P)
            odds_ratio = upper_odds / lower_odds
            new_df = pd.DataFrame(odds_ratio).T
            new_df.index = [i]
            if i == 0:
                shuffle_dict[col] = new_df.copy()
            else:
                shuffle_dict[col] = pd.concat([shuffle_dict[col], new_df])
    with open(f"{temp_shuffle_folder}/shuffled_odds_ratio_{iteration}.pkl", "wb") as f:
        pickle.dump(shuffle_dict, f)


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="100G", partition="ncpu"),
)
def collate_all_shuffles_upper_lower(temp_shuffle_folder):
    """
    Function to combine the shuffle population tables
    """
    files_to_look = ["shuffled_odds_ratio_"]
    path_to_look = pathlib.Path(temp_shuffle_folder)
    new_folder = path_to_look.parent / "collated_shuffles"
    new_folder.mkdir(parents=True, exist_ok=True)
    cols_to_look = [
        "VISp",
        "VISpor",
        "VISli",
        "VISal",
        "VISl",
        "VISpl",
        "VISpm",
        "VISrl",
        "VISam",
        "VISa",
        "RSPv",
        "RSPd",
        "STR",
        "RSPagl",
        "ACAd",
        "ACAv",
        "SSp",
        "SSs",
        "MOp",
        "MOs",
        "TEa",
        "Contra",
        "AUDd",
        "AUDv",
        "HPF",
        "ECT",
        "PERI",
    ]

    for file_start in files_to_look:
        all_files = path_to_look.glob(f"{file_start}*.pkl")

        concatenated_dfs = {
            col: [] for col in cols_to_look
        }  # Dictionary to hold concatenated DataFrames

        # Process each pickle file
        for f in all_files:
            df_dict = pd.read_pickle(
                f
            )  # Load the dictionary of DataFrames from the pickle file
            for col in cols_to_look:
                if col in df_dict:
                    concatenated_dfs[col].append(
                        df_dict[col]
                    )  # Append the DataFrame to the list

        # Concatenate DataFrames for each column
        for col in cols_to_look:
            if concatenated_dfs[col]:
                concatenated_dfs[col] = pd.concat(concatenated_dfs[col])

        # Save the concatenated DataFrames dictionary back to a new pickle file
        output_file = new_folder / f"{file_start}collated.pkl"
        with open(output_file, "wb") as out_f:
            pd.to_pickle(concatenated_dfs, out_f)

        # Clean up old files
        list_all = os.listdir(path_to_look)
        for file_path in list_all:
            if file_path.startswith(file_start):
                os.remove(path_to_look / file_path)


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
    #     download_allen = pathlib.Path(
    #     "/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/Allen_Connectivity"
    # )
    parameters = load_parameters(directory=parameters_path)
    lcm_dir = parameters["lcm_directory"]
    ROI_3D = np.load(f"{lcm_dir}/ROI_3D_25.npy")
    # these expts have already been downloaded. If not you'll have to re-download
    finalpix_expt_a = pd.read_pickle("mouse_connectivity/finalpix_expt_a.pkl")
    finalpix_expt_b = pd.read_pickle("mouse_connectivity/finalpix_expt_b.pkl")
    finalpix_expt_c = pd.read_pickle("mouse_connectivity/finalpix_expt_c.pkl")
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
	input_matrix_b = input_matrix if num_cols >= num_rows else np.transpose(input_matrix)
	for r in range(iters):
		hp.append(list(np.where(input_matrix_b[r] == 1)[0]))
	return hp


def curve_ball(input_matrix, r_hp, num_iterations=-1):
	num_rows, num_cols = input_matrix.shape
	l = range(len(r_hp))
	num_iters = 5*min(num_rows, num_cols) if num_iterations == -1 else num_iterations
	for rep in range(num_iters):
		AB = sample(l, 2)
		a = AB[0]
		b = AB[1]
		ab = set(r_hp[a])&set(r_hp[b]) # common elements
		l_ab=len(ab)
		l_a=len(r_hp[a])
		l_b=len(r_hp[b])
		if l_ab not in [l_a,l_b]:		
			tot=list(set(r_hp[a]+r_hp[b])-ab)
			ab=list(ab)	
			shuffle(tot)
			L=l_a-l_ab
			r_hp[a] = ab+tot[:L]
			r_hp[b] = ab+tot[L:]
	out_mat = np.zeros(input_matrix.shape, dtype='int8') if num_cols >= num_rows else np.zeros(input_matrix.T.shape, dtype='int8')
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
            if col == area:
                conditional_prob.loc[col, area] = np.nan
            else:
                conditional_prob.loc[col, area] = (
                    matrix[matrix[col] > 0].astype(bool).astype(int)[area].mean()
                )
    return conditional_prob

def get_cosine_sim_of_probs(matrix, cols):
    cosine_sim_matrix = pd.DataFrame(data=np.zeros((len(cols), len(cols))), columns= cols, index=cols)
    for col in cols:
        for col_2 in cols:
            neurons_1 = matrix.loc[col]
            neurons_2 = matrix.loc[col_2]
            neurons_1 =neurons_1.drop([col, col_2])
            neurons_2 =neurons_2.drop([col, col_2])
            bl = np.array(neurons_1).reshape(1, -1)
            bl_2 = np.array(neurons_2).reshape(1, -1)
            cosine_sim = cosine_similarity(bl, bl_2)
            cosine_sim_matrix.loc[col, col_2] = cosine_sim[0][0]
            cosine_sim_matrix.loc[col_2, col] = cosine_sim[0][0]
    np.fill_diagonal(cosine_sim_matrix.values, np.nan)
    return cosine_sim_matrix
