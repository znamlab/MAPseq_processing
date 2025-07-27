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
from lcm_registration import lcm_registration_functions as lrf
from lcm_registration import visualign_functions as vis
from znamutils import slurm_it
from PIL import Image
import os
import yaml
import shutil
from skimage.morphology import binary_closing


def convert_tif_to_jpg(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.startswith("s"):
            filename = "S" + filename[1:]
        if filename.lower().endswith((".tif", ".tiff")):
            try:
                with Image.open(input_path) as img:
                    output_path = os.path.join(
                        output_folder, os.path.splitext(filename)[0] + ".jpg"
                    )
                    img.convert("RGB").save(output_path, "JPEG")

            except OSError as e:
                print(f"Error processing {filename}: {e}")
                continue
        elif filename.lower().endswith(".jpg"):
            output_path = os.path.join(output_folder, filename)
            try:
                shutil.copy(input_path, output_path)
            except OSError as e:
                print(f"Error copying {filename}: {e}")
                continue


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
        parameters_file = pathlib.Path(__file__).parent / "parameters.yml"
    else:
        parameters_file = pathlib.Path(directory) / "parameters.yml"
    with open(parameters_file, "r") as f:
        parameters = flatten_dict(yaml.safe_load(f))
    return parameters


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="20G", partition="ncpu"),
)
def convert_images(parameters_path, overwrite="no"):
    """
    Function to convert LCM images into npy files with non-linear deformation
    Args:
        parameters_path(str): path to where parameters file is (make sure you've inputed lcm registration params)
        overwrite (str): 'yes' or 'no' - whether you want to overwrite current converted images in directory
    Returns:
        None
    """
    # make saving path directory if doesn't exist already
    parameters = load_parameters(directory=parameters_path)
    lcm_aligned_dir = pathlib.Path(parameters["lcm_aligned_dir"])
    saving_path = pathlib.Path(lcm_aligned_dir).parents[1] / "allenccf/allen_ccf_coord"
    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
    with open(lcm_aligned_dir) as fp:
        bla = json.load(fp)
    slice_coord = pd.DataFrame(
        columns=[
            "filename",
            "ox",
            "oy",
            "oz",
            "ux",
            "uy",
            "uz",
            "vx",
            "vy",
            "vz",
            "markers",
            "height",
            "width",
        ],
        dtype=int,
    )
    for slice in bla["slices"]:
        anchoring = slice["anchoring"]
        slice_coord = slice_coord.append(
            {
                "filename": slice["filename"],
                "ox": (anchoring[0]),
                "oy": (anchoring[1]),
                "oz": (anchoring[2]),
                "ux": (anchoring[3]),
                "uy": (anchoring[4]),
                "uz": (anchoring[5]),
                "vx": (anchoring[6]),
                "vy": (anchoring[7]),
                "vz": (anchoring[8]),
                "markers": (slice["markers"]),
                "height": (slice["height"]),
                "width": (slice["width"]),
            },
            ignore_index=True,
        )
    job_list = []
    for i, row in slice_coord["filename"].iteritems():
        if row.endswith(".jpeg"):
            section = row[: -len(".jpeg")]
        elif row.endswith(".jpg"):
            section = row[: -len(".jpg")]
        if section.startswith("_"):
            filename = f"{str(saving_path)}/allen_ccf_converted{section}"
        else:
            filename = f"{str(saving_path)}/allen_ccf_converted_{section}"
        if os.path.exists(f"{filename}.npy") and overwrite == "no":
            print(f"{filename} exists already, moving to next", flush=True)
        else:
            print(
                f"Performing non-linear deformation for {section} at {datetime.datetime.now()}",
                flush=True,
            )
            job = lrf.convert_job(
                section_to_look=section,
                which_one=i,
                parameters_path=parameters_path,
                scripts_name=f"converting_{section}",
                slurm_folder="/camp/home/turnerb/slurm_logs",
                use_slurm="True",
            )
            job_list.append(job)

    job_list = ":".join(map(str, job_list))
    lrf.get_euclidean_distance(
        parameters_path=parameters_path,
        use_slurm=True,
        job_dependency=job_list,
        slurm_folder="/camp/home/turnerb/slurm_logs",
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="20G", partition="ncpu"),
)
def convert_job(section_to_look, parameters_path, which_one):
    """
    Function to convert image into allen ccf
    """
    parameters = load_parameters(directory=parameters_path)
    lcm_aligned_dir = pathlib.Path(parameters["lcm_aligned_dir"])
    saving_path = pathlib.Path(lcm_aligned_dir).parents[1] / "allenccf/allen_ccf_coord"
    if section_to_look.startswith("_"):
        filename = f"{str(saving_path)}/allen_ccf_converted{section_to_look}"
    else:
        filename = f"{str(saving_path)}/allen_ccf_converted_{section_to_look}"
    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
    with open(lcm_aligned_dir) as fp:
        bla = json.load(fp)
    slice_coord = pd.DataFrame(
        columns=[
            "filename",
            "ox",
            "oy",
            "oz",
            "ux",
            "uy",
            "uz",
            "vx",
            "vy",
            "vz",
            "markers",
            "height",
            "width",
        ],
        dtype=int,
    )
    for slice in bla["slices"]:
        anchoring = slice["anchoring"]
        slice_coord = slice_coord.append(
            {
                "filename": slice["filename"],
                "ox": (anchoring[0]),
                "oy": (anchoring[1]),
                "oz": (anchoring[2]),
                "ux": (anchoring[3]),
                "uy": (anchoring[4]),
                "uz": (anchoring[5]),
                "vx": (anchoring[6]),
                "vy": (anchoring[7]),
                "vz": (anchoring[8]),
                "markers": (slice["markers"]),
                "height": (slice["height"]),
                "width": (slice["width"]),
            },
            ignore_index=True,
        )

    allen_matrix_conv = [
        [0, 0, 25, 0],
        [-25, 0, 0, 0],
        [0, -25, 0, 0],
        [13175, 7975, 0, 1],
    ]
    which = slice_coord.iloc[which_one]
    x_val = list(range(0, which["width"]))
    y_val = list(range(0, which["height"]))
    coord = np.meshgrid(x_val, y_val)
    width = which["width"]
    height = which["height"]
    newcoord = []
    # perform non-linear deformation of coordinates on each set of section image pixels according info in json file.
    triangulation = vis.triangulate(width, height, which["markers"])
    for x, y in np.nditer(coord):
        try:
            i, j = vis.transform(triangulation, x, y)
        except TypeError:
            print(f"something wrong at x= {x} y = {y}", Flush=True)
        nc = (i, j)
        newcoord.append(nc)
    # make  new x y matrices of containing new non-linearly deformed coordinates
    gi = pd.DataFrame(newcoord)
    Xt = np.reshape(np.array(gi[0]), (height, width))
    Yt = np.reshape(np.array(gi[1]), (height, width))

    # now transform the deformed and registered quickNII image section coordinates to allen ccf
    print(
        "Converting to Allen Coordinates. %s"
        % datetime.datetime.now().strftime("%H:%M:%S"),
        flush=True,
    )
    U_V_O_vector = [
        [which["ux"], which["uy"], which["uz"]],
        [which["vx"], which["vy"], which["vz"]],
        [which["ox"], which["oy"], which["oz"]],
    ]
    # generate 3D voxels from pixel coordinates for each file
    div_h = (
        which["height"] - 1
    )  # minus one, since u and v vectors vary between 0 and one, and pixels start at 0 (so max Xt/width is 1 only when -1)
    div_w = which["width"] - 1  # ibid
    [xv, yv, zv] = np.matmul([(Xt / div_w), (Yt / div_h), 1], U_V_O_vector)
    # transform into allen coord
    [xa, ya, za, one] = np.matmul([xv, yv, zv, 1], allen_matrix_conv)
    allen_vox = [xa, ya, za, one]
    np.save(filename, allen_vox)


def get_z_value(parameters_path, euclidean):
    """
    Function to get z value for each section, so you can calculate volumes.
    Args:
        parameters_path(str): path to where parameters yaml file is
        lcm_dir (str): where parent LCM directory is
        euclidean (str): true if you want to calculate z from median z difference between nearest 2d pixels in previous sections, 'no' if just want to crudely take average of subtraction of z between sections
        s: 'upper' or 'lower'. Whether the start of the section 's{num}' is capitalised or not. e.g. s001 or S001
    Returns:
        table containing z projections
    """
    # iterate through individual slices, take average difference in in coordinates in z (which is x axes in allen ccf) for last slice (slice s001 for brain 1), take average of previous slices
    parameters = load_parameters(directory=parameters_path)
    lcm_dir = pathlib.Path(parameters["lcm_directory"])
    s = parameters["s"]
    add_z = pd.DataFrame(columns=["slice", "amountz"], dtype=int)
    # need to change for mega thick last bit of cortex section, so extend ROI through 3slices
    saving_path = pathlib.Path(lcm_dir) / "allenccf/z_calc"
    allen_ccf_path = pathlib.Path(lcm_dir) / "allenccf/allen_ccf_coord"
    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
    sections_with_nothing_before = []
    if s == "upper":
        section_start = "S"
    else:
        section_start = "s"
    for file in os.listdir(allen_ccf_path):
        if file.startswith("allen_ccf_converted_"):
            slice_name = file[20:24]
            slicenum = int(file[21:24])
            slice_before = slicenum + 1
            if slice_before > 9:
                slicebefore_name = f"{section_start}0{slice_before}"
            if slice_before < 10:
                slicebefore_name = f"{section_start}00{slice_before}"
            [x1a, y1a, z1a, one1] = np.load(allen_ccf_path / file)
            if pathlib.Path(
                allen_ccf_path / f"allen_ccf_converted_{slicebefore_name}.npy"
            ).exists():
                if euclidean == False:
                    [x1a, y1a, z1a, one1] = np.load(allen_ccf_path / file)
                    [x2a, y2a, z2a, one2] = np.load(
                        allen_ccf_path / f"allen_ccf_converted_{slicebefore_name}.npy"
                    )
                    dif = np.median(x2a.flatten()) - np.median(x1a.flatten())
                    add_z = add_z.append(
                        {"slice": slice_name, "amountz": dif}, ignore_index=True
                    )
                if euclidean == True:
                    eucl_dist = np.load(
                        saving_path / f"euclid_distance_{slice_name}.npy"
                    )
                    z_dist = np.load(saving_path / f"z_add_{slice_name}.npy")
                    # now find the median z_distances for top 20% of the array with the lowest euclidean distance
                    flattened_indices = np.argsort(eucl_dist, axis=None)
                    num_lowest_20_percent = int(len(flattened_indices) * 0.2)
                    lowest_indices = np.unravel_index(
                        flattened_indices[:num_lowest_20_percent], eucl_dist.shape
                    )
                    dif = np.mean(z_dist[lowest_indices])
                    add_z = add_z.append(
                        {"slice": slice_name, "amountz": dif}, ignore_index=True
                    )
            else:
                sections_with_nothing_before.append(slice_name)
    # for slices where the one's before are missing, extend them by the mean of slice z extensions for the others
    median_z = add_z["amountz"].median()
    for slice in sections_with_nothing_before:
        add_z = add_z.append({"slice": slice, "amountz": median_z}, ignore_index=True)
    return add_z


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="5G", partition="ncpu"),
)
def get_euclidean_distance(parameters_path):
    """function to find z distance between slices
    Args:
        parameters_path (str): path to where lcm parameters yml file is
    Returns:
        None
    """
    # add_z = pd.DataFrame(columns=['slice', 'amountz'], dtype=int)
    parameters = load_parameters(directory=parameters_path)
    directory = pathlib.Path(parameters["lcm_directory"])
    saving_path = pathlib.Path(directory) / "allenccf/z_calc"
    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
    allen_ccf_path = pathlib.Path(directory) / "allenccf/allen_ccf_coord"
    sections_with_nothing_before = []
    # sections_for_job = []
    job_list = []
    for file in os.listdir(allen_ccf_path):
        if file.startswith("allen_ccf_converted_"):
            slicenum = int(file[21:24])
            slice_before = slicenum + 1
            slice_name = file[20:24]
            s = file[20]  # specified, as sometimes this is saved in different cases
            if slice_before > 9:
                slicebefore_name = f"{s}0{slice_before}"
            if slice_before < 10:
                slicebefore_name = f"{s}00{slice_before}"
            if pathlib.Path(
                allen_ccf_path / f"allen_ccf_converted_{slicebefore_name}.npy"
            ).exists():
                print(f"sending job for {slice_name}", flush=True)
                job = calc_euclidean_distance(
                    directory=str(directory),
                    slice=slice_name,
                    use_slurm=True,
                    scripts_name=f"z_{slice_name}",
                    slurm_folder="/camp/home/turnerb/slurm_logs",
                )
                job_list.append(job)
            if not pathlib.Path(
                allen_ccf_path / f"allen_ccf_converted_{slicebefore_name}.npy"
            ).exists():
                sections_with_nothing_before.append(slice_name)
    # sections_with_nothing_before = np.array(sections_with_nothing_before)
    # np.save(f"{saving_path}/sections_with_nothing_before", sections_with_nothing_before)
    job_list = ":".join(map(str, job_list))
    lrf.group_ROI_coordinates(
        parameters_path=parameters_path,
        resolution=10,
        run_next="yes",
        use_slurm=True,
        slurm_folder="/camp/home/turnerb/slurm_logs",
        job_dependency=job_list,
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="72:00:00", mem="50G", partition="ncpu"),
)
def calc_euclidean_distance(directory, slice):
    """function to find z distance between slices
    Args:
        directory (str): directory where sections are
        slice (str): which section you're looking at e.g. 's001'
    Returns:
        None
    """
    saving_path = pathlib.Path(directory) / "allenccf/z_calc"
    allen_ccf_path = pathlib.Path(directory) / "allenccf/allen_ccf_coord"
    slicenum = int(slice[1:4])
    slice_before = slicenum + 1
    s = slice[0]
    if slice_before > 9:
        slicebefore_name = f"{s}0{slice_before}"
    if slice_before < 10:
        slicebefore_name = f"{s}00{slice_before}"
    [x1a, y1a, z1a, one1] = np.load(allen_ccf_path / f"allen_ccf_converted_{slice}.npy")
    [x2a, y2a, z2a, one2] = np.load(
        allen_ccf_path / f"allen_ccf_converted_{slicebefore_name}.npy"
    )
    point_z = x1a.flatten()
    points_xy = np.column_stack((z1a.flatten(), y1a.flatten()))
    points_xy1 = np.column_stack((z2a.flatten(), y2a.flatten()))
    point_z2 = x2a.flatten()
    # find the closest point in x1, y1 for each point in x, y
    z_dist_points = []
    closest_dist_points = []
    for i, point in enumerate(points_xy):
        if i % 100 == 0:
            distances = np.linalg.norm(points_xy1 - point, axis=1)
            closest_index = np.argmin(distances)
            z_distance = point_z2[closest_index] - point_z[i]
            z_dist_points.append(z_distance)
            closest_dist_points.append(distances[closest_index])
    closest_dist_points = np.array(closest_dist_points)
    z_dist_points = np.array(z_dist_points)
    # z_dist_points = z_dist_points.reshape(x1a.shape)
    # closest_dist_points = closest_dist_points.reshape(x1a.shape)
    np.save(f"{saving_path}/z_add_{slice}.npy", z_dist_points)
    np.save(f"{saving_path}/euclid_distance_{slice}.npy", closest_dist_points)

    print(f"finished {slice}", flush=True)


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="50G", partition="ncpu"),
)
def group_ROI_coordinates(parameters_path, run_next=yes, resolution=25):
    """
    Function to take a template of allen atlas, then put all the roi's in appropriate coordinate positions, numbered by the sample number.
    Args:
        lcm_directory (str): path to where lcm directory is
        resolution (int): 25um is automatic. 10 or 25 um depending on what resolution you want your 3D numpy array. NB, you need to adjust annotation to pixel if you want to use 10um, or change the annotation nrrd file resolution. I've taken out the option as all registration is done in 25um resolution,so 10um is superfluous.
        run_next(str): 'yes' or 'no', whether you want to run the next functions or not
    Returns:
        None
    """
    parameters = load_parameters(directory=parameters_path)
    lcm_directory = parameters["lcm_directory"]
    annotation_data = nrrd.read(parameters["allen_annotation_path"])
    allen_anno = np.array(annotation_data)
    annotation = allen_anno[0]
    add_z = get_z_value(
        parameters_path=parameters_path, euclidean=parameters["euclidean"]
    )
    if resolution == 10:
        empty_frame = np.zeros(
            (1320, 800, 1140)
        )  # this is the shape of the average template 10um ccf
    elif resolution == 25:
        empty_frame = np.zeros((528, 320, 456))
    print("starting", flush=True)
    ROI_path = pathlib.Path(lcm_directory) / "rois"
    reg_dir = pathlib.Path(lcm_directory) / "allenccf/allen_ccf_coord"
    s = parameters["s"]
    if s == "upper":
        section_start = "S"
    else:
        section_start = "s"
    for ROI_to_look in os.listdir(ROI_path):
        region = ROI_path / ROI_to_look
        if ROI_to_look.startswith("s0") or ROI_to_look.startswith("S0"):
            slicename = region.stem[1:4]
            tube = int(region.stem[5 : len(region.stem)].split("TUBE", 1)[1])
            # some of the roi's are pooled due to missing lcm images that are too hard to determine, these are specified in parameters file. We group these into one sample
            for tube_to_group in parameters["rois_to_combine"]:
                if tube in parameters["rois_to_combine"][tube_to_group]:
                    tube = tube_to_group
            # if int(tube) in cortical_samples_table['Tube'].to_list():
            [xa, ya, za, one] = np.load(
                reg_dir / f"allen_ccf_converted_{section_start}{slicename}.npy"
            )
            roi = plt.imread(ROI_path / f"{region}")
            allencoord_roiya = roi * ya
            allencoord_roiza = roi * za
            allencoord_roixa = roi * xa
            z_to_add = add_z.loc[
                add_z["slice"] == f"{section_start}{slicename}", "amountz"
            ].iloc[0]

            # convert the x, y, z coordinates to pixel
            pixcoord = []
            for i, axis in enumerate(
                [allencoord_roixa, allencoord_roiya, allencoord_roiza]
            ):
                pixel = np.array(np.round(axis / resolution), dtype=int)
                pixel[pixel < 0] = 0
                pixel[pixel >= empty_frame.shape[i]] = 0
                pixcoord.append(pixel)
            new_coord = np.zeros(pixcoord[0].shape)
            z_add = 0

            for stack in range(int(np.round(z_to_add / resolution))):
                for i in range(pixcoord[0].shape[0]):
                    for j in range(pixcoord[0].shape[1]):
                        if pixcoord[0][i, j] != 0:
                            new_coord[i, j] = (pixcoord[0][i, j]) + z_add
                z_add = z_add + 1
                for k in range(pixcoord[0].shape[0]):
                    for l in range(pixcoord[0].shape[1]):
                        x = new_coord[k, l]
                        y = pixcoord[1][k, l]
                        z = pixcoord[2][k, l]
                        if x != 0 and y != 0 and z != 0:
                            # don't include ROI regions that are outside the brain
                            if (
                                annotation[int(x), int(y), int(z)] != 0
                            ):  # don't include ROI regions that are outside the brain
                                empty_frame[int(x), int(y), int(z)] = int(tube)
    # now correct registration errors
    updated_roi_array = remove_hemisphere_overlap(empty_frame)
    updated_roi_array = remove_roi_holes(updated_roi_array)
    # because removing the holes actually smoothes out the rois and then adds pixels to the outside of the brain, let's reset so only rois within the brain are annotated
    annotation_mask = np.where(annotation != 0, 1, 0)
    updated_roi_array = updated_roi_array * annotation_mask
    print("finished")
    np.save(f"{lcm_directory}/ROI_3D_{resolution}.npy", updated_roi_array)

    print("finished")
    if run_next == "yes":
        print("finished, sending final job")
        generate_region_table_across_samples(
            parameters_path=parameters_path,
            use_slurm=True,
            slurm_folder="/camp/home/turnerb/slurm_logs",
        )


def remove_hemisphere_overlap(roi_array):
    """
    Function to correct error in registration where separation of cortical hemispheres leads to error where Visualign non-linear
    registation fails and ROIs on border of one hemisphere have some points in other hemisphere
    Args:
        roi_array (numpy array): 3D ROI array from group_ROI_coordinates
    Returns:
        corrected ROI array
    """
    # determine the midpoint of the array along the z-axis (corresponds to the axis connecting hemipshperes)
    x_midpoint = roi_array.shape[2] // 2

    # select ROIs in each hemisphere
    left_hemisphere_roi = roi_array[:, :, :x_midpoint]
    right_hemisphere_roi = roi_array[:, :, x_midpoint:]

    # identify ROIs that are predominantly in one hemisphere
    left_hemisphere_labels = np.unique(left_hemisphere_roi)
    right_hemisphere_labels = np.unique(right_hemisphere_roi)

    majority_left_labels = [
        label
        for label in left_hemisphere_labels
        if np.sum(left_hemisphere_roi == label) > np.sum(right_hemisphere_roi == label)
    ]
    majority_right_labels = [
        label
        for label in right_hemisphere_labels
        if np.sum(right_hemisphere_roi == label) > np.sum(left_hemisphere_roi == label)
    ]

    # remove those crossing hemispheres where the majority of the ROI is in another hemisphere
    left_mask = np.isin(left_hemisphere_roi, majority_left_labels)
    left_hemisphere_roi[~left_mask] = 0
    right_mask = np.isin(right_hemisphere_roi, majority_right_labels)
    right_hemisphere_roi[~right_mask] = 0

    # combine the hemispheres back into a single array
    roi_array_processed = np.concatenate(
        (left_hemisphere_roi, right_hemisphere_roi), axis=2
    )

    return roi_array_processed


def remove_roi_holes(roi_array):
    """
    Function to close holes in ROIs generated by large deformations in nonlinear registration
    """
    structure = np.ones((2, 2, 2))  # structure that we use to close holes
    roi_3d = roi_array.astype("uint8")
    to_attribute = roi_3d == 0
    cleaned_3d = roi_3d.copy()
    for tube in np.unique(roi_3d):
        if tube == 0:
            continue
        mask = roi_3d == tube
        closed_mask = binary_closing(mask, structure)
        cleaned_3d[to_attribute] = closed_mask[to_attribute].astype("uint8") * tube
        to_attribute = cleaned_3d == 0
    return cleaned_3d


def check_non_target(arr, nontarget_ids):
    """function to check if an id in the non target list"""
    for num in nontarget_ids:
        if num in arr:
            return False
    return True


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="48:00:00", mem="50G", partition="ncpu"),
)
def generate_region_table_across_samples(parameters_path):
    """
    Function to take 3D numpy array from 'group_ROI_coordinates' function, and generate a table containing allen atlas index for brain regions in each sample, as well as the total volume
    Args:
        parameters_path(str): path to where directory where parameters yml file is contained.
    Returns:
        None
    """
    parameters = load_parameters(directory=parameters_path)
    lcm_dir = pathlib.Path(parameters["lcm_directory"])
    roi_array = np.load(lcm_dir / "ROI_3D_25.npy")
    annotation_data = nrrd.read(parameters["allen_annotation_path"])
    nontarget_ids = []
    nontarget_list = ["fiber tracts"]
    bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=True)
    for nontarget in nontarget_list:
        nontarget_ids.append(bg_atlas.structures[nontarget]["id"])
    allen_anno = np.array(annotation_data)
    annotation = allen_anno[0]
    roi_numbers = np.unique(roi_array)[1:]
    voxel_volume = 25**3  # voxels are 25um resolution

    roi_volumes = []
    roi_regions = []

    for roi_num in roi_numbers:
        # count voxels for each ROI
        roi_voxel_count = np.sum(roi_array == roi_num)
        # calculate volume in cubic micrometers
        roi_volume_um = roi_voxel_count * voxel_volume
        roi_volumes.append(roi_volume_um)

        # find brain regions corresponding to each voxel in the ROI
        regions = []
        for index in np.argwhere(roi_array == roi_num):
            annotation_label = annotation[tuple(index)]
            if check_non_target(
                arr=bg_atlas.structures[annotation_label]["structure_id_path"],
                nontarget_ids=nontarget_ids,
            ):
                regions.append(annotation_label)

        roi_regions.append(regions)

    region_samples_dataframe = pd.DataFrame(
        {
            "ROI Number": roi_numbers,
            "Volume (um^3)": roi_volumes,
            "Brain Regions": roi_regions,
        }
    )
    region_samples_dataframe.to_pickle(lcm_dir / "sample_vol_and_regions.pkl")


def check_z_dist(lcm_dir, section, fig, axs):
    euclid_dist = np.load(f"{lcm_dir}/allenccf/z_calc/euclid_distance_S0{section}.npy")
    z_dist = np.load(f"{lcm_dir}/allenccf/z_calc/z_add_S0{section}.npy")
    sb.heatmap(euclid_dist, ax=axs[0, 0])
    axs[0, 0].set_title("Nearest Euclidian distance")
    sb.heatmap(z_dist, ax=axs[0, 1])
    axs[0, 1].set_title("Z distance")
    if os.path.isfile(f"{lcm_dir}/sections_same_orientation/S0{section}.TIF"):
        image_to_look = f"{lcm_dir}/sections_same_orientation/S0{section}.TIF"
    elif os.path.isfile(f"{lcm_dir}/sections_same_orientation/S0{section}.tif"):
        image_to_look = f"{lcm_dir}/sections_same_orientation/S0{section}.tif"
    img1 = plt.imread(image_to_look)
    axs[1, 0].imshow(img1)
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Section looked at")
    if os.path.isfile(f"{lcm_dir}/sections_same_orientation/S0{section+1}.TIF"):
        image_to_look_2 = f"{lcm_dir}/sections_same_orientation/S0{section+1}.TIF"
    elif os.path.isfile(f"{lcm_dir}/sections_same_orientation/S0{section+1}.tif"):
        image_to_look_2 = f"{lcm_dir}/sections_same_orientation/S0{section+1}.tif"
    img2 = plt.imread(image_to_look_2)

    axs[1, 1].imshow(img2)
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Section before")
    plt.tight_layout()
