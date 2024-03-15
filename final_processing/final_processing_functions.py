from bg_atlasapi import BrainGlobeAtlas
from preprocessing_sequencing import preprocess_sequences as ps
from znamutils import slurm_it
import pandas as pd
from final_processing import final_processing_functions as fpf
import numpy as np

def find_adjacent_samples(ROI_array, sample_list):
    """
    Function to find adjacent samples surrounding cubelets within 50um distance of max (N.B. you might have to change this since it only extends the maximum coordinates, and if not two cubes it does not detect)
    Args:
        ROI_array: 3D numpy array in 25um resolution
        sample_list: list of samples you want to find adjacent samples for
    Returns:
        Dictionary containing adjacent samples for each sample
    """
    adjacent_dict = {}
    def get_coordinate_bounds(coordinates):
        bounds = []
        for num in range(3):
            min_val = min(coordinates, key=lambda x: x[num])[num]
            max_val = max(coordinates, key=lambda x: x[num])[num]
            bounds.append((min_val, max_val))
        return bounds
    def adjust_coordinates(coordinates, axis, range_axis): #adjust coordinates to +/- 50um the min and max coordinates
        if axis == 0:
            min_coord_adj = [[coord[0]-2, coord[1], coord[2]] for coord in [coord for coord in coordinates if coord[0] == range_axis[0]]]
            max_coord_adj = [[coord[0]+2, coord[1], coord[2]] for coord in [coord for coord in coordinates if coord[0] == range_axis[1]]]
        if axis == 1:
            min_coord_adj = [[coord[0], coord[1]-2, coord[2]] for coord in [coord for coord in coordinates if coord[1] == range_axis[0]]]
            max_coord_adj = [[coord[0], coord[1]+2, coord[2]] for coord in [coord for coord in coordinates if coord[1] == range_axis[1]]]
        if axis == 2:
            min_coord_adj = [[coord[0], coord[1], coord[2]-2] for coord in [coord for coord in coordinates if coord[2] == range_axis[0]]]
            max_coord_adj = [[coord[0], coord[1], coord[2]+2] for coord in [coord for coord in coordinates if coord[2] == range_axis[1]]]
        
        min_coord_adj.extend(max_coord_adj)
        return min_coord_adj

    def get_adjacent_samples(coordinates, array):
        sample_list = []
        for coord in coordinates:
            sample_num = array[coord[0], coord[1], coord[2]]
            if sample_num != 0:
                sample_list.append(sample_num)
        return sample_list

    for ROI_sample in sample_list:
        coordinates = np.argwhere(ROI_array == ROI_sample)
        coord_to_scan = []
        for i, range_axis in enumerate(get_coordinate_bounds(coordinates)):
            coord_from_axis = adjust_coordinates(coordinates=coordinates, axis=i, range_axis=range_axis)
            coord_to_scan.extend(coord_from_axis)
        all_adjacent_samples =get_adjacent_samples(coord_to_scan, ROI_array)
        adjacent_dict[ROI_sample] = np.unique(all_adjacent_samples)
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
    RT_to_sample = pd.read_csv(parameters['RT_to_sample'])
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
    #now remove any samples that have been excluded in parameters yaml
    if parameters['samples_to_drop']:
        list_samples =[x for x in parameters['samples_to_drop'] if x in barcode_table.columns] #amend list if they aren't in barcode table columns, as some are aggregated in rois_to_combine
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
    if id>0:
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
        group_structures = ['HY', 'CB', 'MY', 'P', 'fiber tracts', 'STR', 'IPN', 'BLA', 'PAL', 'HPF', 'SCm', 'SCs', 'IC', 'LGd', 'LGv', 'root', 'SSp', 'MOB', 'AOB']
        olfactory_bulb = ['MOB', 'AOB']
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
            newid = 'OB'
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
    #first let's define the A1 source sites by taking the barcodes that have max with min amount A1
    parameters = ps.load_parameters(directory=parameters_path)
    sample_vol_and_regions_table = pd.read_pickle(sample_vol)
    sample_vol_and_regions_table['regions'] = 'NA'
    sample_vol_and_regions_table['breakdown'] = 'NA'
    sample_vol_and_regions_table['vol_in_atlas'] = 0
    sample_vol_and_regions_table['main'] = 'NA'
    sample_vol_and_regions_table['main_fraction'] = 0
    
    for index, row in sample_vol_and_regions_table.iterrows():
        all_regions = sample_vol_and_regions_table.loc[index]['Brain Regions']
        all_regions= [i for i in all_regions if i != 0]
        all_reg_converted = []
        all_reg, counts = np.unique(all_regions, return_counts=True)
        for i in all_reg:
            converted = fpf.get_id(i)
            all_reg_converted.append(converted)
        if row['ROI Number'] in parameters['contra']:
            all_reg_converted = ['Contra-' + s for s in all_reg_converted]
        region_counts = pd.DataFrame({'Regions': all_reg_converted, 'Counts': counts})
        sum_values = region_counts.groupby('Regions').sum()
        sum_values = sum_values.sort_values(by='Counts', ascending=False, ignore_index=False)
        sum_values['Fraction'] = sum_values['Counts']/sum_values['Counts'].sum()
        sample_vol_and_regions_table.loc[index, 'regions'] = str(sum_values.index.to_list())
        sample_vol_and_regions_table.loc[index, 'breakdown'] = str(sum_values.Fraction.to_list())
        sample_vol_and_regions_table.loc[index, 'vol_in_atlas'] = sum_values['Counts'].sum()
        sample_vol_and_regions_table.loc[index, 'main'] = sum_values.iloc[0].name
        sample_vol_and_regions_table.loc[index, 'main_fraction'] = sum_values.iloc[0].Fraction
    sample_vol_and_regions_table.to_pickle(sample_vol)