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


def convert_tif_to_jpg(input_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_folder):
        # Check if the file is a TIFF image
        if filename.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(input_folder, filename)

            try:
                # Load the TIFF image
                with Image.open(input_path) as img:
                    # Construct the output file path
                    output_path = os.path.join(
                        output_folder, os.path.splitext(filename)[0] + ".jpg"
                    )

                    # Save as JPEG
                    img.convert("RGB").save(output_path, "JPEG")

            except OSError as e:
                print(f"Error processing {filename}: {e}")
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
    slurm_options=dict(ntasks=1, time="24:00:00", mem="350G", partition="hmem"),
)
def convert_images(parameters_path, overwrite="yes"):
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
    # incorporate allen conversion units, and subsequently also incorporate functions from NITRC.org
    allen_matrix_conv = [
        [0, 0, 25, 0],
        [-25, 0, 0, 0],
        [0, -25, 0, 0],
        [13175, 7975, 0, 1],
    ]
    for i, row in slice_coord["filename"].iteritems():
        if row.endswith(".jpeg"):
            section = row[: -len(".jpeg")]
        elif row.endswith(".jpg"):
            section = row[: -len(".jpg")]
        filename = f"{str(saving_path)}/allen_ccf_converted_{section}"
        if os.path.exists(f"{filename}.npy") and overwrite == "no":
            print(f"{filename} exists already, moving to next", flush=True)
        else:
            print(
                f"Performing non-linear deformation for {section} at {datetime.datetime.now()}",
                flush=True,
            )
            which = slice_coord.iloc[i]
            x_val = list(range(0, which["width"]))
            y_val = list(range(0, which["height"]))
            coord = np.meshgrid(x_val, y_val)
            width = which["width"]
            height = which["height"]
            newcoord = []
            # perform non-linear deformation of coordinates on each set of section image pixels according info in json file.
            triangulation = vis.triangulate(width, height, which["markers"])
            for x, y in np.nditer(coord):
                i, j = vis.transform(triangulation, x, y)
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
    slurm_options=dict(ntasks=1, time="24:00:00", mem="5G", partition="cpu"),
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
        use_slurm=True,
        slurm_folder="/camp/home/turnerb/slurm_logs",
        job_dependency=job_list,
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="72:00:00", mem="50G", partition="cpu"),
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
    slurm_options=dict(ntasks=1, time="48:00:00", mem="50G", partition="cpu"),
)
def get_roi_vol(parameters_path):
    """
    Function to calculate roi volumes.
    Args:
        parameters_path(str): path to where parameters yml file is
        lcm_dir (str): parent directory for lcm reg
        allen_anno_path (str): path to where allen annotation nrrd file is
        s: whether the start of the section 's{num}' is capitalised or not. e.g. s001 or S001
    Returns:
        None
    """
    # load annotation
    parameters = load_parameters(directory=parameters_path)
    allen_anno_path = parameters["allen_annotation_path"]
    s = parameters["s"]
    lcm_dir = pathlib.Path(parameters["lcm_directory"])
    add_z = get_z_value(
        parameters_path=parameters_path, euclidean=parameters["euclidean"]
    )
    allen_anno = nrrd.read(allen_anno_path)
    allen_anno = np.array(allen_anno)
    annotation = allen_anno[0]
    roi_path = pathlib.Path(lcm_dir) / "rois"
    ROI_vol = pd.DataFrame()
    if s == "upper":
        section_start = "S"
    else:
        section_start = "s"
    for region in os.listdir(roi_path):
        if region.startswith("S0") or region.startswith("s0"):
            slice_name = f"{section_start}{region[1:4]}"
            tube = region[5 : len(region)].split("TUBE", 1)[1]
            tube = tube[:-4]
            [xa, ya, za, one] = np.load(
                lcm_dir
                / f"allenccf/allen_ccf_coord/allen_ccf_converted_{slice_name}.npy"
            )
            roi = plt.imread(roi_path / f"{region}")
            allencoord_roiya = roi * ya
            allencoord_roiza = roi * za
            # use shoelace formula to define area of polygon given xy coordinates then calculate volume of each LCM roi
            calcz = allencoord_roiza[allencoord_roiza != 0]
            calcy = allencoord_roiya[allencoord_roiya != 0]
            area_roi = 0.5 * np.abs(
                np.dot(calcz, np.roll(calcy, 1)) - np.dot(calcy, np.roll(calcz, 1))
            )
            z_to_add = add_z.loc[add_z["slice"] == slice_name, "amountz"].iloc[0]
            if (
                z_to_add > 0
            ):  # the sign of z changes, depending which direction you're measuring it from
                vol_roi = area_roi * z_to_add
            else:
                vol_roi = area_roi * -z_to_add
            # convert the x, y, z coordinates to pixel
            pixcoord = []
            for i, axis in enumerate([xa, ya, za]):
                pixel = np.array(np.round(axis / 25), dtype=int)
                pixel[pixel < 0] = 0
                pixel[pixel >= annotation.shape[i]] = 0
                pixcoord.append(pixel)

            # use annotation.json to convert each pixel to region id

            registered_slice = np.zeros(xa.shape, dtype=annotation.dtype)
            a2 = annotation[
                pixcoord[0].flatten(), pixcoord[1].flatten(), pixcoord[2].flatten()
            ].reshape(registered_slice.shape)
            ROI_anno = a2 * roi
            # iterate image by z slices, each additional z, annotate then add to list
            if z_to_add > 0:
                slices = round(z_to_add / 25)
            else:
                slices = -round(z_to_add / 25)
            for x in range(slices):
                if x > 0:
                    newz = (
                        pixcoord[0] + x
                    )  # changed from plus to minus as going backwards
                    slice = annotation[
                        newz.flatten(), pixcoord[1].flatten(), pixcoord[2].flatten()
                    ].reshape(registered_slice.shape)
                    ROI_anno_add = slice * roi
                    ROI_anno = np.append(ROI_anno, ROI_anno_add)

            unique, counts = np.unique(ROI_anno, return_counts=True)
            region_vol = (counts / sum(counts)) * vol_roi
            ROI_vol = ROI_vol.append(
                {
                    "slice": slice_name,
                    "tube": tube,
                    "z_added": z_to_add,
                    "vol (um3)": vol_roi,
                    "region_pix": ROI_anno,
                    "unique_regions": unique[1:],
                    "region_vol (um3)": region_vol[1:],
                },
                ignore_index=True,
            )
    ROI_vol.to_pickle(lcm_dir / "ROI_vol.pkl")


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

    # combine volumes for LCM
    ROI_vol = pd.read_pickle(ROI_vol_path)
    final_pix = pd.DataFrame(columns=["tube", "combined_pix", "vol (um3)"], dtype=int)
    result = ROI_vol.groupby(["tube"]).agg(", ".join).reset_index()
    for row, tube in result["tube"].iteritems():
        newdf = ROI_vol[ROI_vol["tube"] == tube].reset_index()
        # for count, value in enumerate(newdf):
        for r, t in newdf["tube"].iteritems():
            if r == 0:
                array = newdf.loc[r, "region_pix"]
                vol = newdf.loc[r, "vol (um3)"]
            if r > 0:
                nextarray = newdf.loc[r, "region_pix"]
                vol = vol + newdf.loc[r, "vol (um3)"]
                array = np.concatenate((array, nextarray), axis=None)
        final_pix = final_pix.append(
            {"tube": tube, "combined_pix": array, "vol (um3)": vol}, ignore_index=True
        )
    # generate list of unique id regions in all samples
    for r, tube in final_pix["tube"].iteritems():
        if r == 0:
            array = final_pix.loc[r, "combined_pix"]
        if r > 0:
            next_array = final_pix.loc[r, "combined_pix"]
            array = np.concatenate((array, next_array), axis=None)
    all_regions = np.unique(array)
    region_col = all_regions[all_regions != 0]
    # #calculate region volume in each tube, then create a heatmap of regions coloured according to region
    final_pix.tube = final_pix.tube.astype(float)
    finalpix1 = final_pix.sort_values("tube").reset_index()
    all_regions = np.unique(array)
    np.save(str(f"{lcm_dir}/region_col.npy"), region_col)
    finalpix1.to_pickle(f"{lcm_dir}/finalpix.pkl")
    # final_pix.to_pickle(lcm_dir/"finalpix.pkl")


def get_acronymn(lcm_dir):
    """
    Function to take annotations, and get the acronymn for each of the brain areas that the ROIs are in
    Args:
        lcm_dir (str): the directory where the lcm registration info is all in
    Returns:
        None
    """
    # now generate empty table for the acronymns of all areas based on allen ccf
    acronymncol = []
    for id in regioncol:
        if bg_atlas.structures[id]["acronym"][-1].isnumeric():
            newid = bg_atlas.structures[id]["structure_id_path"][
                -2
            ]  # moving one level up the hierarchy if cortical layer
        elif (
            bg_atlas.structures[id]["acronym"][-2:] == "6a"
            or bg_atlas.structures[id]["acronym"][-2:] == "6b"
        ):
            newid = bg_atlas.structures[id]["structure_id_path"][
                -2
            ]  # moving one level up the hierarchy if layer 6a/6b
        else:
            newid = id
        acronymn = bg_atlas.structures[newid]["acronym"]
        acronymncol.append(acronymn)
    acronymncol = np.unique(acronymncol).tolist()
    region_table = pd.DataFrame(columns=acronymncol, dtype=int)
    # need to generate reference table to convert id's into higher bit of hierarchy.

    for row, tube in finalpix1["tube"].iteritems():
        regions = finalpix1.loc[row, "combined_pix"]
        unique, counts = np.unique(regions, return_counts=True)
        region_area = (counts / sum(counts)) * (finalpix1.loc[row, "vol (um3)"])
        regions = unique[1:]
        region_area = region_area[1:]
        values = regions, region_area
        region_table.at[row, "sample"] = tube
        index = -1
        if regions.size != 0:
            for id in np.nditer(regions):
                index += 1
                if bg_atlas.structures[id]["acronym"][-1].isnumeric():
                    newid = bg_atlas.structures[id]["structure_id_path"][
                        -2
                    ]  # moving one level up the hierarchy if cortical layer
                elif (
                    bg_atlas.structures[id]["acronym"][-2:] == "6a"
                    or bg_atlas.structures[id]["acronym"][-2:] == "6b"
                ):
                    newid = bg_atlas.structures[id]["structure_id_path"][
                        -2
                    ]  # moving one level up the hierarchy if layer 6a/6b
                else:
                    newid = id
                acronym = bg_atlas.structures[newid]["acronym"]
                region_table.at[row, acronym] = region_area[index]
    region_tab_contra = region_table
    # take areas in samples of contralateral hemisphere, and re-label as belonging to contra

    for i, row in region_table.iterrows():
        if region_table["sample"].iloc[i] in contra_samples:
            for col in region_table.columns:
                if (
                    col != "sample"
                    and col.startswith("Contra") == False
                    and np.isnan(region_table[col].iloc[i]) == False
                ):
                    newcol = "Contra-" + col
                    if newcol not in region_tab_contra:
                        region_tab_contra[newcol] = 0
                    region_tab_contra[newcol].iloc[i] = region_table[col].iloc[i]
                    region_tab_contra[col].iloc[i] = 0

    nozero = region_table.fillna(0)


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="24:00:00", mem="350G", partition="hmem"),
)
def group_ROI_coordinates(parameters_path):
    """
    Function to take a template of allen atlas, then put all the roi's in appropriate coordinate positions, numbered by the sample number.
    Args:
        lcm_directory (str): path to where lcm directory is
    Returns:
        None
    """
    parameters = load_parameters(directory=parameters_path)
    lcm_directory = parameters["lcm_directory"]
    add_z = get_z_value(
        parameters_path=parameters_path, euclidean=parameters["euclidean"]
    )
    empty_frame = np.zeros(
        (1320, 800, 1140)
    )  # this is the shape of the average template 10um ccf
    ROI_path = pathlib.Path(lcm_directory) / "rois"
    reg_dir = pathlib.Path(lcm_directory) / "allenccf/allen_ccf_coord"
    s = parameters["s"]
    if s == "upper":
        section_start = "S"
    else:
        section_start = "s"
    for ROI_to_look in os.listdir(ROI_path):
        # region = ROI_path/'s015_TUBE6.png'
        region = ROI_path / ROI_to_look
        if ROI_to_look.startswith("s0") or ROI_to_look.startswith("S0"):
            slicename = region.stem[1:4]
            tube = region.stem[5 : len(region.stem)].split("TUBE", 1)[1]
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
                pixel = np.array(np.round(axis / 10), dtype=int)
                pixel[pixel < 0] = 0
                pixel[pixel >= empty_frame.shape[i]] = 0
                pixcoord.append(pixel)
            new_coord = np.zeros(pixcoord[0].shape)
            z_add = 0

            for stack in range(int(np.round(z_to_add / 10))):
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
                            empty_frame[int(x), int(y), int(z)] = int(tube)

    remove_hemisphere_overlap(empty_frame)
    np.save(f"{lcm_directory}/ROI_3D.npy", empty_frame)

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

    # Combine the hemispheres back into a single array
    roi_array_processed = np.concatenate(
        (left_hemisphere_roi, right_hemisphere_roi), axis=2
    )

    return roi_array_processed


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="48:00:00", mem="50G", partition="cpu"),
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
    roi_array = np.load(lcm_dir / "ROI_3D.npy")
    annotation_data = nrrd.read(parameters["allen_annotation_path"])
    allen_anno = np.array(annotation_data)
    annotation = allen_anno[0]
    roi_numbers = np.unique(roi_array)[1:]
    voxel_volume = (
        10**3
    )  # Voxel volume in cubic micrometers (assuming 10um resolution)

    roi_volumes = []
    roi_regions = []

    for roi_num in roi_numbers:
        # Count voxels for each ROI
        roi_voxel_count = np.sum(roi_array == roi_num)
        # Calculate volume in cubic micrometers
        roi_volume_um = roi_voxel_count * voxel_volume
        roi_volumes.append(roi_volume_um)

        # Find brain regions corresponding to each voxel in the ROI
        regions = []
        for index in np.argwhere(roi_array == roi_num):
            annotation_label = annotation[tuple(index)]
            regions.append(annotation_label)

        roi_regions.append(regions)
    region_samples_dataframe = pd.DataFrame(
        {
            "ROI Number": roi_numbers,
            "Volume (um^3)": roi_volumes,
            "Brain Regions": roi_regions,
        }
    )
    region_samples_dataframe.to_pickle(lcm_dir / "sample_vol_and_regions")
