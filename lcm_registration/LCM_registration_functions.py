import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import json
import os
import datetime
import nrrd
import bg_space as bg
from pprint import pprint
from matplotlib.colors import LogNorm, Normalize
import seaborn as sb
import pathlib
import pathlib
from bg_atlasapi import BrainGlobeAtlas
from lcm_registration import LCM_registration_functions as lrf
from lcm_registration import visualign_functions as vis
from znamutils import slurm_it

def load_parameters(directory="root"):
    """Load the parameters yaml file containting all the parameters required for
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
        parameters_file = pathlib.Path(__file__).parent / "lcm_parameters.yml"
    else:
        parameters_file = pathlib.Path(directory) / "lcm_parameters.yml"
    with open(parameters_file, "r") as f:
        parameters = flatten_dict(yaml.safe_load(f))
    return parameters


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="350G", partition="hmem"),
)
def convert_images(lcm_aligned_dir):
    """
    Function to convert LCM images into npy files with non-linear deformation
    Args:
        lcm_aligned_dir(str): path to where json file where Visualign output is
    Returns: 
        None
     """
    #make saving path directory if doesn't exist already
    saving_path = pathlib.Path(lcm_aligned_dir).parents[1]/'allenccf/allen_ccf_coord'
    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
    with open(lcm_aligned_dir) as fp:
        bla = json.load(fp)
    slice_coord = pd.DataFrame(columns=['filename','ox', 'oy', 'oz', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz', 'markers', 'height', 'width'], dtype=int)
    for slice in bla["slices"]:
        anchoring = slice["anchoring"]
        slice_coord = slice_coord.append({'filename': slice["filename"], 'ox': (anchoring[0]), 'oy': (anchoring[1]), 'oz': (anchoring[2]), 'ux': (anchoring[3]), 'uy': (anchoring[4]), 'uz': (anchoring[5]), 'vx': (anchoring[6]), 'vy': (anchoring[7]), 'vz': (anchoring[8]), 'markers': (slice["markers"]),'height': (slice["height"]), 'width': (slice["width"])},ignore_index=True)
    #incorporate allen conversion units, and subsequently also incorporate functions from NITRC.org
    allen_matrix_conv = [[0, 0, 25, 0],
                        [-25, 0, 0, 0],
                        [0, -25, 0, 0],
                        [13175, 7975, 0, 1]]
    for i, row in slice_coord['filename'].iteritems():
        section =row[:-len('.jpeg')]
        filename = f'{str(saving_path)}/allen_ccf_converted{section}'
        if os.path.exists(f'{filename}.npy'):
            print(f'{filename} exists already, moving to next', flush=True)
        else:
            print(f'Performing non-linear deformation for {section} at {datetime.datetime.now()}', flush=True)
            which = slice_coord.iloc[i]
            x_val = list(range(0, which['width']))
            y_val = list(range(0, which['height']))
            coord = np.meshgrid(x_val, y_val)
            width=which['width']
            height=which['height']
            newcoord=[]
        #perform non-linear deformation of coordinates on each set of section image pixels according info in json file.
            triangulation=vis.triangulate(width,height,which["markers"])
            for x, y in np.nditer(coord):
                i,j=vis.transform(triangulation,x,y)
                nc = (i,j)
                newcoord.append(nc)
        #make  new x y matrices of containing new non-linearly deformed coordinates
            gi = pd.DataFrame(newcoord)
            Xt = np.reshape(np.array(gi[0]), (height, width))
            Yt = np.reshape(np.array(gi[1]), (height, width))

        #now transform the deformed and registered quickNII image section coordinates to allen ccf
            print('Converting to Allen Coordinates. %s' %
                    datetime.datetime.now().strftime('%H:%M:%S'), flush=True)
            U_V_O_vector = [[which['ux'], which['uy'], which['uz']],
                            [which['vx'], which['vy'], which['vz']],
                            [which['ox'], which['oy'], which['oz']]]
        #generate 3D voxels from pixel coordinates for each file
            div_h = which['height']-1 #minus one, since u and v vectors vary between 0 and one, and pixels start at 0 (so max Xt/width is 1 only when -1)
            div_w = which['width']-1 #ibid
            [xv,  yv,  zv] = np.matmul([(Xt/div_w),  (Yt/div_h),  1 ], U_V_O_vector)
        #transform into allen coord
            [xa, ya, za, one] = np.matmul([xv,  yv,  zv, 1], allen_matrix_conv)
            allen_vox = [xa, ya, za, one]
            np.save(filename, allen_vox)

def get_z_value(lcm_dir, OB_first):
    """
    Function to get z value for each section, so you can calculate volumes.
    Args:
        lcm_dir (str): where parent LCM directory is
        OB_first (str): 'yes' or 'no' - whether slicing direction is from OB to cerebellum or not 
        sections_with_nothing_before (list): list of sections (e.g.  ['s001', 's005']) whether there isn't a section in front of to take the z projection from
    Returns:
        table containing z projections
    """
    #iterate through individual slices, take average difference in in coordinates in z (which is x axes in allen ccf) for last slice (slice s001 for brain 1), take average of previous slices
    add_z = pd.DataFrame(columns=['slice', 'amountz'], dtype=int)
    #need to change for mega thick last bit of cortex section, so extend ROI through 3slices
    saving_path = pathlib.Path(lcm_dir)/'allenccf/allen_ccf_coord'
    sections_with_nothing_before = []
    for file in os.listdir(saving_path):
        if file.startswith('allen_ccf_converted_'):
            slice_name = file[20:24]
            slicenum = int(file[21:24])
           # if OB_first == 'yes':
            #    slice_before= slicenum-1
           # elif OB_first == 'no':
            slice_before= slicenum+1
            if slice_before >9:
                slicebefore_name = f's0{slice_before}'
            if slice_before<10:
                slicebefore_name = f's00{slice_before}' 
            [x1a, y1a, z1a, one1] = np.load(saving_path/file)
            if pathlib.Path(saving_path/f'allen_ccf_converted_{slicebefore_name}.npy').exists():
                [x1a, y1a, z1a, one1] = np.load(saving_path/file)
                [x2a, y2a, z2a, one2] = np.load(saving_path/f'allen_ccf_converted_{slicebefore_name}.npy')
                dif = np.average(x2a.flatten()-x1a.flatten())
                add_z= add_z.append({'slice': slice_name, 'amountz': dif},ignore_index=True)
            else:
                sections_with_nothing_before.append(slice_name)
    #for slices where the one's before are missing, extend them by the mean of slice z extensions for the others
    average_z = add_z['amountz'].mean()
    for slice in sections_with_nothing_before:
        add_z= add_z.append({'slice': slice, 'amountz': average_z},ignore_index=True)   
    return add_z
    
def get_roi_vol(lcm_dir, add_z, allen_anno_path):
    """
    Function to calculate roi volumes.
    Args:
        lcm_dir (str): parent directory for lcm reg
        add_z: table output from 'get_z_value' function
        allen_anno_path (str): path to where allen annotation nrrd file is
    Returns:
        ROI_vol table    
    """
    #load annotation
    allen_anno = nrrd.read(allen_anno_path)
    allen_anno = np.array(allen_anno)
    annotation = allen_anno[0]
    roi_path = pathlib.Path(lcm_dir)/'rois'
    ROI_vol=pd.DataFrame()
    for region in os.listdir(roi_path):
        if region.startswith("S0") or region.startswith("s0"):
            slice_name = f's{region[1:4]}'
            tube = region[5:len(region)].split('TUBE', 1)[1]
            tube =tube[:-4]
            [xa, ya, za, one] = np.load(lcm_dir/f'allenccf/allen_ccf_coord/allen_ccf_converted_{slice_name}.npy')
            roi = plt.imread(roi_path / f'{region}')
            allencoord_roiya = roi*ya
            allencoord_roiza = roi*za
    #use shoelace formula to define area of polygon given xy coordinates then calculate volume of each LCM roi
            calcz = allencoord_roiza[allencoord_roiza!= 0]
            calcy = allencoord_roiya[allencoord_roiya!= 0]
            area_roi = 0.5*np.abs(np.dot(calcz,np.roll(calcy,1))-np.dot(calcy,np.roll(calcz,1)))
            z_to_add = add_z.loc[add_z['slice'] == slice_name, 'amountz'].iloc[0]
            if z_to_add > 0: #the sign of z changes, depending which direction you're measuring it from
                vol_roi = area_roi*z_to_add
            else:
                vol_roi = area_roi*-z_to_add
    #convert the x, y, z coordinates to pixel
            pixcoord = []
            for i, axis in enumerate([xa, ya, za]):
                pixel = np.array(np.round(axis/25), dtype=int)
                pixel[pixel <0] = 0
                pixel[pixel >= annotation.shape[i]] = 0
                pixcoord.append(pixel)

    # use annotation.json to convert each pixel to region id

            registered_slice = np.zeros(xa.shape, dtype = annotation.dtype)
            a2=annotation[pixcoord[0].flatten(),
                        pixcoord[1].flatten(),
                        pixcoord[2].flatten()].reshape(registered_slice.shape)
            ROI_anno = a2*roi
    #iterate image by z slices, each additional z, annotate then add to list
            if z_to_add > 0:
                slices = round(z_to_add/25)
            else:
                slices = -round(z_to_add/25)
            for x in range(slices):
                if x >0:
                    if z_moving_towards_OB == 'yes':
                        newz = pixcoord[0]+x
                    else:
                        newz = pixcoord[0]-x #changed from plus to minus as going backwards
                    slice = annotation[newz.flatten(), pixcoord[1].flatten(), pixcoord[2].flatten()].reshape(registered_slice.shape)
                    ROI_anno_add = slice*roi
                    ROI_anno = np.append(ROI_anno, ROI_anno_add)
        
            unique, counts = np.unique(ROI_anno, return_counts=True)
            region_vol = (counts/sum(counts))*vol_roi
            ROI_vol= ROI_vol.append({'slice': slice_name, 'tube': tube, 'z_added': z_to_add, 'vol (um3)': vol_roi, 'region_pix': ROI_anno, 'unique_regions': unique[1:], 'region_vol (um3)': region_vol[1:]},ignore_index=True)
    ROI_vol.to_pickle(lcm_dir/"ROI_vol.pkl")
    return ROI_vol

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="350G", partition="hmem"),
)
def combine_tubes(lcm_dir, ROI_vol_path):
    """
    Function to combine all the LCM roi's that have been combined in the same tube
    Args:
        lcm_dir (str): path to where parent lcm registration directory is
        ROI_vol_path: path to table output of 'get_roi_vol' function
    Returns:
        None
    """
    #combine volumes for LCM
    ROI_vol = pd.read_pickle(ROI_vol_path)
    final_pix = pd.DataFrame(columns=['tube', 'combined_pix', 'vol (um3)'], dtype=int)
    result = ROI_vol.groupby(['tube']).agg(', '.join).reset_index()
    for row, tube in result['tube'].iteritems():
        newdf = ROI_vol[ROI_vol['tube']==tube].reset_index()
        #for count, value in enumerate(newdf):
        for r, t in newdf['tube'].iteritems():
            if r ==0 :
                array = newdf.loc[r, 'region_pix']
                vol = newdf.loc[r, 'vol (um3)']
            if r > 0:
                nextarray = newdf.loc[r, 'region_pix']
                vol = vol + newdf.loc[r, 'vol (um3)']
                array = np.concatenate((array, nextarray), axis=None)
        final_pix= final_pix.append({'tube': tube, 'combined_pix': array, 'vol (um3)': vol},ignore_index=True)
    #generate list of unique id regions in all samples
    for r, tube in final_pix['tube'].iteritems():
        if r ==0 :
            array = final_pix.loc[r, 'combined_pix']
        if r > 0:
            next_array = final_pix.loc[r, 'combined_pix']
            array = np.concatenate((array, next_array), axis=None)
    all_regions= np.unique(array)
    region_col =  all_regions[all_regions!= 0]
    # #calculate region volume in each tube, then create a heatmap of regions coloured according to region
    final_pix.tube = final_pix.tube.astype(float)
    finalpix1 =final_pix.sort_values('tube').reset_index()
    all_regions= np.unique(array)
    np.save(str(f'{lcm_dir}/region_col.npy'), region_col)
    finalpix1.to_pickle(f"{lcm_dir}/finalpix.pkl")
    #final_pix.to_pickle(lcm_dir/"finalpix.pkl")
    
