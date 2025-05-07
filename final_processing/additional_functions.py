import yaml
import pathlib
import pandas as pd
import ast
import numpy as np

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

def normalize_barcodes(barcodes):
    #since we have the soma barcode here, we normalise to 1 for the second max (non-som, max projection) value, then set the soma to 1
    normalized_barcodes = barcodes.copy().astype(float)
    for index, row in normalized_barcodes.iterrows():
        max_val = row.max()  # Find the maximum value
        # Exclude the maximum value from normalization
        other_values_sum = row.sum() - max_val
        if other_values_sum > 0:  # Avoid division by zero
            normalized_barcodes.loc[index] = row / other_values_sum
        normalized_barcodes.loc[index][row.idxmax()] = 1.0  # Set max value to 1
    return normalized_barcodes

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
        lcm_directory=pathlib.Path(lcm_directory),
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
        barcodes = send_for_curveball_shuff(barcodes = barcodes) #note this is only in final processing functions script
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
# def get_area_volumes(barcode_table_cols, lcm_directory, area_threshold=0.1):
#     """
#     Function to get volumes of each registered brain area from each LCM sample
#     Args:
#         barcode_table_cols: list of column names of the barcode matrix
#         lcm_directory: path to where the lcm_directory is
#         area_threshold (int): minimum value that defines a brain area within a cubelet
#     Returns: area vol pandas dataframe
#     """
#     sample_vol_and_regions = pd.read_pickle(
#         "".join([lcm_directory, "/sample_vol_and_regions.pkl"])
#     )
#     sample_vol_and_regions = sample_vol_and_regions[
#         sample_vol_and_regions["ROI Number"].isin(barcode_table_cols)
#     ]
#     sample_vol_and_regions["regions"] = sample_vol_and_regions["regions"].apply(
#         ast.literal_eval
#     )
#     sample_vol_and_regions["breakdown"] = sample_vol_and_regions["breakdown"].apply(
#         ast.literal_eval
#     )
#     all_areas_unique_acronymn = np.unique(
#         sample_vol_and_regions["regions"].explode().to_list()
#     )
#     all_area_df = pd.DataFrame(
#         index=barcode_table_cols, columns=all_areas_unique_acronymn
#     )
#     for column in barcode_table_cols:
#         # all_regions = sample_vol_and_regions_FIAA456d.loc[sample_vol_and_regions_FIAA456d.index[sample_vol_and_regions_FIAA456d['ROI Number'] == column].tolist(), 'Brain Regions'].explode().astype(int)
#         index = sample_vol_and_regions[
#             sample_vol_and_regions["ROI Number"] == column
#         ].index
#         reg = pd.DataFrame()
#         reg["regions"] = [i for i in sample_vol_and_regions.loc[index, "regions"]][0]
#         reg["fraction"] = [i for i in sample_vol_and_regions.loc[index, "breakdown"]][0]
#         reg["vol_area"] = (
#             reg["fraction"] * sample_vol_and_regions.loc[index, "Volume (um^3)"].item()
#         )

#         for _, row in reg.iterrows():
#             all_area_df.loc[column, row["regions"]] = row["vol_area"]
#     group_areas = {"Contra": all_area_df.filter(like="Contra").columns}
#     areas_grouped = all_area_df.copy()
#     for group, columns in group_areas.items():
#         areas_grouped[group] = areas_grouped.filter(items=columns).sum(axis=1)
#         columns = [value for value in columns if value in all_area_df.columns]
#         areas_grouped = areas_grouped.drop(columns, axis=1)
#     nontarget_list = ["fiber tracts", "root"]
#     nontarget_list = [value for value in nontarget_list if value in all_area_df.columns]
#     areas_only_grouped = areas_grouped.drop(nontarget_list, axis=1)
#     areas_only_grouped = areas_only_grouped.apply(
#         lambda row: row.where(row >= area_threshold * row.sum(), np.nan), axis=1
#     )
#     areas_only_grouped = areas_only_grouped.fillna(0)
#     areas_only_grouped = areas_only_grouped.loc[
#         :, (areas_only_grouped != 0).any(axis=0)
#     ]

#     return areas_only_grouped
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
def find_non_overlapping_position(ax, x, y, label, placed_texts, renderer, jitter_range=10, max_tries=50):
    for _ in range(max_tries):
        collision = False
        test_text = ax.text(x, y, label, color="white", fontsize=8, ha='center', va='center', alpha=0)  # Invisible text for testing
        bb_new = test_text.get_window_extent(renderer=renderer)

        for existing_text in placed_texts:
            bb_existing = existing_text.get_window_extent(renderer=renderer)
            if bb_existing.overlaps(bb_new):
                collision = True
                break

        test_text.remove()  # Remove test text
        if not collision:
            return x, y
        else:
            # Apply jitter to avoid overlap
            x += np.random.uniform(-jitter_range, jitter_range)
            y += np.random.uniform(-jitter_range, jitter_range)

    return x, y  # Return the best position found (may still overlap if max_tries is exceeded)