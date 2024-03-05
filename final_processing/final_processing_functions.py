from bg_atlasapi import BrainGlobeAtlas
from preprocessing_sequencing import preprocess_sequences as ps
from znamutils import slurm_it
import pandas as pd

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
        group_structures = ['HY', 'CB', 'MY', 'P', 'fiber tracts', 'OLF', 'STR', 'IPN', 'BLA', 'PAL', 'HPF', 'SCm', 'SCs', 'IC', 'LGd', 'LGv', 'root']
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
        
        # ancestors= bg_atlas.get_structure_ancestors(id)
        # newid = [item for item in group_structures if item in ancestors]
        return newid
    
@slurm_it(
conda_env="MAPseq_processing",
module_list=None,
slurm_options=dict(ntasks=1, time="24:00:00", mem="10G", partition="cpu"),
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
    reg_names = []
    main_region = []
    for i, row in sample_vol_and_regions_table.iterrows():
        all_regions = sample_vol_and_regions_table.loc[i]['Brain Regions']
        all_regions= [i for i in all_regions if i != 0]
        all_reg_converted = []
        for i in all_regions:
            converted = get_id(i)
            all_reg_converted.append(converted)
        regions, counts = np.unique(all_reg_converted, return_counts=True)
        region_counts = pd.DataFrame({'Regions': regions, 'Counts': counts}).sort_values(by='Counts', ascending=False, ignore_index=True)
        first_amount = np.round(region_counts.loc[0]['Counts']/region_counts['Counts'].sum(), 2)
        first_region = region_counts.loc[0]['Regions']
        if r['ROI number'] in parameters['contra']:
                first_region= f'contra-{first_region}'
        if len(regions)>1:
            second_region = region_counts.loc[1]['Regions']
            second_amount = np.round(region_counts.loc[1]['Counts']/region_counts['Counts'].sum(), 2)
            if r['ROI number'] in parameters['contra']:
                second_region = f'contra-{second_region}'
            breakdown = f'{first_amount} {first_region} & {second_amount} {second_region}'
        else:
            breakdown = f'{first_amount} {first_region}'
        reg_names.append(breakdown)
        main_region.append(first_region)
    sample_vol_and_regions_table['reg_names'] = reg_names
    sample_vol_and_regions_table['main_region'] = main_region
    sample_vol_and_regions_table.to_pickle(sample_vol)