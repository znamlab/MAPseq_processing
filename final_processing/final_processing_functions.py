from bg_atlasapi import BrainGlobeAtlas
from preprocessing_sequencing import preprocess_sequences as ps
from znamutils import slurm_it
import pandas as pd
from final_processing import final_processing_functions as fpf
import numpy as np
import nrrd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import pathlib
import ast
from sklearn.linear_model import LinearRegression, Lasso


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
    parameters = ps.load_parameters(directory=parameters_path)
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
    parameters = ps.load_parameters(directory=parameters_path)
    RT_to_sample = pd.read_csv(parameters["RT_to_sample"])
    RT_to_sample.set_index("sample", inplace=True)
    mapping_barcode_table = RT_to_sample["tube"].to_dict()
    barcode_table.rename(columns=mapping_barcode_table, inplace=True)
    # drop the non-existant tubes, added so that there wasn't gaps in RT to sample
    barcode_table = barcode_table.drop(0, axis=1)
    for tube_to_group in parameters["rois_to_combine"]:
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
    bg_atlas = BrainGlobeAtlas("allen_mouse_10um", check_latest=False)
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
    slurm_options=dict(ntasks=1, time="24:00:00", mem="50G", partition="cpu"),
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
    parameters = ps.load_parameters(directory=parameters_path)
    sample_vol_and_regions_table = pd.read_pickle(sample_vol)
    sample_vol_and_regions_table["regions"] = "NA"
    sample_vol_and_regions_table["breakdown"] = "NA"
    # sample_vol_and_regions_table['vol_in_atlas'] = 0
    sample_vol_and_regions_table["main"] = "NA"
    sample_vol_and_regions_table["main_fraction"] = 0

    for index, row in sample_vol_and_regions_table.iterrows():
        all_regions = sample_vol_and_regions_table.loc[index]["Brain Regions"]
        all_regions = [i for i in all_regions if i != 0]
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


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="48:00:00", mem="350G", partition="hmem"),
)
def calculate_strength_GP_regression(parameters_path, shuffle):
    """Function to take barcode matrix, and with the assumption that single neuron projection patterns are spatially smooth, use gaussian process regression to
    map projection strengh across different areas. N.B. you need to have a 2D ROI flatmap npy saved. Since this requires python 3.9, and different environment,
    Run from notebook 'create_2D_cortical_flatmap' with MAPseq_processing_py39 environment.
    Args:
        parameters_path(str): path to where parameters file is
        shuffle (bool): whether to shuffle columns in barcode table or not as a negative control

    Returns:
        None
    """
    parameters = ps.load_parameters(directory=parameters_path)
    mouse = parameters["MOUSE"]
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
        sequencing_directory / "A1_barcodes.pkl"  # this has source samples removed
    )
    if shuffle:
        barcodes = send_to_shuffle(barcodes=barcodes_across_sample.to_numpy())
        barcodes_across_sample = pd.DataFrame(data=barcodes, columns=barcodes_across_sample.columns)
        barcodes_across_sample.to_pickle(sequencing_directory / 'A1_barcodes_shuffled.pkl')
    
    lcm_directory = parameters["lcm_directory"]
    ROI_2D = np.load(f"{lcm_directory}/cortical_flatmap.npy")
    # remove tubes in ROI flatmap that aren't in normalised barcode path

    cortical_samples = parameters["cortical_samples"]
    cortical_samples = np.array(cortical_samples)
    cortical_samples = cortical_samples[np.isin(cortical_samples, np.unique(ROI_2D))]
    # since we've removed the source sites, we also might want cortical samples that are source sites removed
    cortical_samples = [
        i for i in cortical_samples if i in barcodes_across_sample.columns
    ]
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample[cortical_samples].astype(bool).sum(axis=1) > 1
    ].reset_index(drop=True)
    to_process = barcodes_across_sample.index
    chunk_size = 25
    splits = np.array_split(to_process, len(to_process) // chunk_size)
    pathlib.Path(f"{parameters_path}/temp").mkdir(parents=True, exist_ok=True)
    if shuffle:
        mouse = f'{mouse}_shuffled'
    job_ids = []
    for i in range(len(splits)):
        job_id = calculate_strength_GP_regression_chunk(
            parameters_path=parameters_path,
            chunk_indices=list(splits[i]), shuffle=shuffle,
            num=i,
            use_slurm=True,
            scripts_name=f"GP_reg_chunk_{i}_{mouse}",
            slurm_folder=parameters["SLURM_DIR"],
        )
        job_ids.append(job_id)
    job_ids = ",".join(map(str, job_ids))
    collate_chunks(
        parameters_path=parameters_path, shuffle=shuffle,
        use_slurm=True, 
        scripts_name=f"collate_{mouse}",
        slurm_folder=parameters["SLURM_DIR"],
        job_dependency=job_ids,
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="6:00:00", mem="20G", partition="ncpu"),
)
def calculate_strength_GP_regression_chunk(parameters_path, chunk_indices, shuffle, num):
    """
    Function to process individual chunks from calculate GP regression function
    """
    parameters = ps.load_parameters(directory=parameters_path)
    lcm_directory = parameters["lcm_directory"]
    mouse = parameters["MOUSE"]
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
    if shuffle == False:
        barcodes_across_sample = pd.read_pickle(
        sequencing_directory / "A1_barcodes.pkl"  # this has source samples removed
    )
    elif shuffle == True:
        barcodes_across_sample = pd.read_pickle(
        sequencing_directory / "A1_barcodes_shuffled.pkl"  
    )
    
    
    ROI_2D = np.load(f"{lcm_directory}/cortical_flatmap.npy")
    # remove tubes in ROI flatmap that aren't in normalised barcode path
    cortical_samples = parameters["cortical_samples"]
    cortical_samples = np.array(cortical_samples)
    cortical_samples = cortical_samples[np.isin(cortical_samples, np.unique(ROI_2D))]
    # since we've removed the source sites, we also might want cortical samples that are source sites removed
    cortical_samples = [
        i for i in cortical_samples if i in barcodes_across_sample.columns
    ]
    centroids = []
    for sample in cortical_samples:
        centroids.append(np.argwhere(ROI_2D == sample).mean(axis=0))
    
    centroids = np.stack(centroids)

    temp_folder = f"{parameters_path}/temp"
    
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample[cortical_samples].astype(bool).sum(axis=1) > 1
    ].reset_index(drop=True)
    
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample.index.isin(chunk_indices)
    ]
    sample_matrix = ROI_2D.T
   
   
    tubes = np.arange(
        np.min(barcodes_across_sample.columns),
        np.max(barcodes_across_sample.columns),
        1,
    )
    tubes_not_in = [
        i for i in tubes if i not in barcodes_across_sample.columns.to_list()
    ]
    for x in tubes_not_in:
        ROI_2D[ROI_2D == x] = 0
    barcode_matrix = np.zeros(
        (
            len(barcodes_across_sample),
            max(barcodes_across_sample.columns.astype(int)) + 1,
        )
    )
    for column in barcodes_across_sample:
        barcode_matrix[:, int(column)] = barcodes_across_sample[column].to_numpy()
    labels_df = pd.read_csv(
        "/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/LCM_registration/labelDescription_ITKSNAPColor.txt",
        header=None,
        sep="\s+",
        index_col=0,
    )
    labels_df.columns = ["r", "g", "b", "x0", "x1", "x2", "acronym"]

    annotation_data = nrrd.read(
        "/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/LCM_registration/flatmap_butterfly.nrrd"
    )
    allen_anno = np.array(annotation_data)
    annotation = allen_anno[0]
    flipped = np.flip(annotation.T, 1)
    combined = np.hstack((annotation.T[:, :1176], flipped[:, 184:]))

    areas_in_flatmap = [
        labels_df.loc[index, "acronym"]
        for index in labels_df.index
        if index in combined
    ]
    contra_areas = [f"Contra-{x}" for x in areas_in_flatmap]
    areas_in_flatmap.extend(contra_areas)
    all_barcode_projections = pd.DataFrame(columns=areas_in_flatmap)
    kernel = WhiteKernel() + Matern(length_scale=10, length_scale_bounds=(30, 100))
    for index_to_look_neuron in range(len(barcode_matrix)):
        # kernel = WhiteKernel() + Matern(length_scale=10, length_scale_bounds=(50, 150))
        y = barcode_matrix[index_to_look_neuron, cortical_samples]
        # soma_idx = np.argmax(y) somamax is already removed from A1 dataset, therefore not necessary to take out
        gpr = GaussianProcessRegressor(
            kernel=kernel, random_state=0, n_restarts_optimizer=5
        ).fit(centroids, y)
        ycoor, xcoor = np.mgrid[0 : sample_matrix.shape[0], 0 : sample_matrix.shape[1]]
        X = np.concatenate((xcoor.reshape((-1, 1)), ycoor.reshape((-1, 1))), axis=1)
        pred = gpr.predict(X)
        area_dict = {}
        barcode_2D = np.reshape(pred, sample_matrix.shape) / pred.sum()
        for area in areas_in_flatmap:
            mask = np.zeros_like(combined, dtype=bool)
            rows, cols = combined.shape
            if area.startswith("Contra-"):
                acro = area[len("Contra-") :]
                index_to_look = labels_df[labels_df["acronym"] == acro].index.to_list()
                mask[:, : cols // 2] = combined[:, : cols // 2] == index_to_look
                selected_values = barcode_2D[mask]
            else:
                index_to_look = labels_df[labels_df["acronym"] == area].index.to_list()
                mask[:, cols // 2 :] = combined[:, cols // 2 :] == index_to_look
                selected_values = barcode_2D[mask]
            average_counts = np.mean(selected_values)
            if average_counts < 0:
                average_counts = 0
            average_counts_per_um = average_counts/(np.sum(mask)*25)
            area_dict[area] = average_counts_per_um
        all_barcode_projections = all_barcode_projections.append(
            area_dict, ignore_index=True
        )
    if shuffle:
        mouse = f'{mouse}_shuffled'
    all_barcode_projections.to_pickle(
        f"{temp_folder}/GP_regression_projections_{mouse}_{num}.pkl"
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="6:00:00", mem="8G", partition="ncpu"),
)
def collate_chunks(parameters_path, shuffle):
    """
    Function to collate all the tables from GP regression and save combined
    """
    parameters = ps.load_parameters(directory=parameters_path)
    mouse = parameters["MOUSE"]
    temp_folder = pathlib.Path(f"{parameters_path}/temp")
    name = 'GP_regression_projections_collated'
    if shuffle:
        mouse = f'{mouse}_shuffled'
        name = f'{name}_shuffled'
    all_files = temp_folder.glob(f"GP_regression_projections_{mouse}_*.pkl")
    all_tables = []
    for f in all_files:
        all_tables.append(pd.read_pickle(f))
    all_tables = pd.concat(all_tables)
    all_tables.to_pickle(f"{parameters_path}/{name}.pkl")


def homog_across_cubelet_cortical(parameters_path, cortical, shuffled):
    """
    Function to output a matrix of homogenous across areas, looking only at cortical samples
    Args:
        parameters_path
        cortical (bool): True if you want onkly to look at cortical regions
        shuffled (bool): True if you want to shuffle values in all columns as a negative control
    """
    parameters = ps.load_parameters(directory=parameters_path)
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
    barcodes_across_sample = pd.read_pickle(sequencing_directory / "A1_barcodes.pkl")
    lcm_directory = parameters["lcm_directory"]
    # sample_vol_and_regions = pd.read_pickle(''.join([lcm_directory, '/sample_vol_and_regions.pkl']))
    # sample_vol_and_regions['regions'] = sample_vol_and_regions['regions'].apply(ast.literal_eval)
    # sample_vol_and_regions['breakdown'] = sample_vol_and_regions['breakdown'].apply(ast.literal_eval)
    # all_areas_unique_acronymn = np.unique(sample_vol_and_regions['regions'].explode().to_list())
    cortical_samples_columns = [
        int(col)
        for col in parameters["cortical_samples"]
        if col in barcodes_across_sample.columns
    ]
    # only look at cortical samples
    if cortical:
        barcodes_across_sample = barcodes_across_sample[cortical_samples_columns]
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample.astype(bool).sum(axis=1) > 1
    ]
    # all_area_df = pd.DataFrame(index = barcodes_across_sample.columns, columns=all_areas_unique_acronymn)
    # for column in barcodes_across_sample.columns:
    #     #all_regions = sample_vol_and_regions_FIAA456d.loc[sample_vol_and_regions_FIAA456d.index[sample_vol_and_regions_FIAA456d['ROI Number'] == column].tolist(), 'Brain Regions'].explode().astype(int)
    #     index = sample_vol_and_regions[sample_vol_and_regions['ROI Number']==column].index
    #     reg = pd.DataFrame()
    #     reg['regions'] = [i for i in sample_vol_and_regions.loc[index, 'regions']][0]
    #     reg['fraction'] = [i for i in sample_vol_and_regions.loc[index, 'breakdown']][0]
    #     reg['volume'] = reg['fraction']*sample_vol_and_regions.loc[index, 'Volume (um^3)'].item()

    #     for _, row in reg.iterrows():
    #         all_area_df.loc[column, row['regions']] = row['volume']
    # combine contra
    # areas_grouped = all_area_df.copy()
    # col = areas_grouped.filter(like="Contra")
    # areas_grouped['Contra'] = col.sum(axis=1)
    # areas_grouped =areas_grouped.drop(columns=col.columns)
    # nontarget_list = ['fiber tracts', 'root']
    # nontarget_list = [value for value in nontarget_list if value in all_area_df.columns]
    # areas_only_grouped = areas_grouped.drop(nontarget_list, axis=1)
    # create a dataframe of the fractions of each brain area contained within each sample
    # areas_only_grouped = areas_only_grouped.fillna(0)
    areas_only_grouped = get_area_volumes(
        barcode_table_cols=barcodes_across_sample.columns, lcm_directory=lcm_directory
    )
    areas_matrix = areas_only_grouped.to_numpy()
    total_frac = np.sum(areas_matrix, axis=1)
    frac_matrix = areas_matrix / total_frac[:, np.newaxis]
    weighted_frac_matrix = frac_matrix / areas_matrix.sum(axis=0)
    # barcodes_sum = barcodes_across_sample.sum(axis=1)
    # barcodes_across_sample =barcodes_across_sample.div(barcodes_sum, axis=0)
    barcodes = barcodes_across_sample.to_numpy()
    if shuffled:
        barcodes = send_to_shuffle(barcodes=barcodes)
    # areasFrac = pd.DataFrame(frac_matrix, columns=areas_only_grouped.columns)
    # for each barcode, create a matrix of BC count for regions in a sample based on amount of each region in LCM (makes assumption of equal BC distribution)
    # frac_matrix = all_area_fractions_FIAA456a.to_numpy()
    # bc_matrix = np.zeros(shape=((len(barcodes_across_sample), (len(areas_only_grouped.columns)))))
    # bc_matrix = pd.DataFrame(data= bc_matrix, columns=areas_only_grouped.columns, index=barcodes_across_sample.index)
    # for i, row in barcodes_across_sample.iterrows():
    #     counts = row.to_numpy()
    #     frac_counts = frac_matrix * counts[:, np.newaxis]
    #     sample_counts = pd.DataFrame(frac_counts, columns=areas_only_grouped.columns)
    #     bc_matrix.loc[i] = sample_counts.sum() / frac_matrix.sum()
    # remove any columns that are all zeros
    # for column in bc_matrix.columns:
    #   if bc_matrix[column].sum() == 0:
    #      bc_matrix.drop([column], axis=1, inplace=True)
    total_projection_strength = np.sum(barcodes, axis=1)
    barcodes /= total_projection_strength[:, np.newaxis]
    bc_matrix = np.matmul(barcodes, weighted_frac_matrix)
    bc_matrix = pd.DataFrame(
        data=bc_matrix, columns=areas_only_grouped.columns.to_list()
    )
    # bool_barcodes= barcodes_across_sample.astype(bool).astype(int).to_numpy()
    # vol_matrix = np.matmul(bool_barcodes, areas_matrix)
    # bc_matrix = bc_matrix/vol_matrix
    bc_matrix = bc_matrix.loc[~(bc_matrix == 0).all(axis=1)]
    # bc_matrix =bc_matrix.reset_index(drop=True)
    return bc_matrix.fillna(0)


def get_area_volumes(barcode_table_cols, lcm_directory):
    """
    Function to get volumes of each registered brain area from each LCM sample
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
    return areas_only_grouped.fillna(0)


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


def homog_across_area_cortical(parameters_path, cortical, shuffled):
    """
    Function to output a matrix of homogenous across areas, looking only at cortical samples
    Args:
        parameters_path
        cortical (bool): True if you want onkly to look at cortical regions
        shuffled (bool): True if you want to shuffle values in all columns as a negative control
    """
    parameters = ps.load_parameters(directory=parameters_path)
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
    barcodes_across_sample = pd.read_pickle(sequencing_directory / "A1_barcodes.pkl")
    lcm_directory = parameters["lcm_directory"]

    # sample_vol_and_regions = pd.read_pickle(''.join([lcm_directory, '/sample_vol_and_regions.pkl']))
    # sample_vol_and_regions['regions'] = sample_vol_and_regions['regions'].apply(ast.literal_eval)
    # sample_vol_and_regions['breakdown'] = sample_vol_and_regions['breakdown'].apply(ast.literal_eval)
    # all_areas_unique_acronymn = np.unique(sample_vol_and_regions['regions'].explode().to_list())

    cortical_samples_columns = [
        int(col)
        for col in parameters["cortical_samples"]
        if col in barcodes_across_sample.columns
    ]
    # only look at cortical samples
    if cortical:
        barcodes_across_sample = barcodes_across_sample[cortical_samples_columns]
    barcodes_across_sample = barcodes_across_sample[
        barcodes_across_sample.astype(bool).sum(axis=1) > 1
    ]
    areas_only_grouped = get_area_volumes(
        barcode_table_cols=barcodes_across_sample.columns, lcm_directory=lcm_directory
    )
    # all_area_df = pd.DataFrame(index = barcodes_across_sample.columns, columns=all_areas_unique_acronymn)
    # for column in barcodes_across_sample.columns:
    #     #all_regions = sample_vol_and_regions_FIAA456d.loc[sample_vol_and_regions_FIAA456d.index[sample_vol_and_regions_FIAA456d['ROI Number'] == column].tolist(), 'Brain Regions'].explode().astype(int)
    #     index = sample_vol_and_regions[sample_vol_and_regions['ROI Number']==column].index
    #     reg = pd.DataFrame()
    #     reg['regions'] = [i for i in sample_vol_and_regions.loc[index, 'regions']][0]
    #     reg['fraction'] = [i for i in sample_vol_and_regions.loc[index, 'breakdown']][0]
    #     reg['vol_area'] = reg['fraction']*sample_vol_and_regions.loc[index, 'Volume (um^3)'].item()

    #     for _, row in reg.iterrows():
    #         all_area_df.loc[column, row['regions']] = row['vol_area']
    # group_areas = {'Contra': all_area_df.filter(like="Contra").columns}
    # areas_grouped = all_area_df.copy()
    # for group, columns in group_areas.items():
    #     areas_grouped[group] = areas_grouped.filter(items=columns).sum(axis=1)
    #     columns = [value for value in columns if value in all_area_df.columns]
    #     areas_grouped = areas_grouped.drop(columns, axis=1)
    # nontarget_list = ['fiber tracts', 'root']
    # nontarget_list = [value for value in nontarget_list if value in all_area_df.columns]
    # areas_only_grouped = areas_grouped.drop(nontarget_list, axis=1)
    # #create a dataframe of the fractions of each brain area contained within each sample
    # areas_only_grouped = areas_only_grouped.fillna(0)
    areas_matrix = areas_only_grouped.to_numpy()
    # total_frac = np.sum(areas_matrix, axis=1)
    # frac_matrix = areas_matrix/total_frac[:, np.newaxis]

    # barcodes_across_sample.fillna(0,inplace=True)
    barcodes_matrix = barcodes_across_sample.to_numpy()
    # barcodes_matrix[np.isnan(barcodes_matrix)] = 0
    if shuffled:
        barcodes_matrix = send_to_shuffle(barcodes=barcodes_matrix)
    total_projection_strength = np.sum(barcodes_matrix, axis=1)
    normalised_bc_matrix = barcodes_matrix / total_projection_strength[:, np.newaxis]
    normalised_bc_matrix = normalised_bc_matrix[
        total_projection_strength > 0, :
    ]  # needed as already removed barcodes with no projections some nan values if shuffled and no projections in some barcodes

    mdl = Lasso(fit_intercept=False, positive=True)
    # mdl = Lasso(fit_intercept=False, positive=True)
    # mdl.fit(frac_matrix, barcodes_matrix.T)
    mdl.fit(areas_matrix, normalised_bc_matrix.T)
    barcodes_homog = pd.DataFrame(mdl.coef_, columns=areas_only_grouped.columns)

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


def organise_GP_reg_results(parameters_path):
    """
    Function to sort columns for analysis of GP reg matrix
    """
    parameters = ps.load_parameters(directory=parameters_path)
    regression_res = pd.read_pickle(
        f"{parameters_path}/GS_regression_projections_collated.pkl"
    )
    contra_cols = regression_res[regression_res.filter(like="Contra").columns].sum(
        axis=1
    )
    regression_res = regression_res.drop(
        columns=regression_res.filter(like="Contra").columns
    )
    regression_res["Contra"] = contra_cols
    SSp = regression_res.filter(like="SSp").sum(axis=1)
    regression_res = regression_res.drop(
        columns=regression_res.filter(like="SSp").columns.to_list()
    )
    regression_res["SSp"] = SSp
    return regression_res
