# MAPseq_processing
Pipeline for processing MAPseq NGS datasets

**Image naming conventions:**
Images of each LCM section are taken on the LCM microscope and aligned to same images used for registration so that LCM regions can be mapped. An overview section of each image must be taken before performing LCM. You must take the image on the side of the section that is most anterior and name each section to end with: "_s[number of section]". Numbering of sections goes from anterior to posterior - e.g. "_s001" for OB, even if you didn't slice this section first. For each lcm image, you must name the image by section first, then tube name e.g. "s001_tube1".

**Generate ROI masks for each LCM sample:**
First update LCM parameters yaml file with directory name, mouse name, project name etc.
Using matlab, run the lcm_registration/Align_LCM_to_section.m matlab script. LCM images are looped through per slice (you need to specify which section you want to do in the script, as well as the directory) and you need to manually draw a polygon ROI in the images that pop up - in red will be highlighted the difference between the image before and after LCM, so you can easily identify where the ROI is. If you accidentally draw one ROI wrong, use lcm_registration/Align_LCM_to_single_section.m so you don't want to loop though all the tubes again. If there are errors in registration due to deformed image/bad stitching go to either Correctingbadstitching.m (where you need to select the badly stitched frames within the image and select a point that is continuous on the slice, do this iteratively if you want multiple frames to be registered, then repeat the Align_LCM_to_section script with this corrected image as input). For Point_alignment.m you just need to select the same regions in each image so that it is manually aligned.

**Register sections to Allen atlas**
Register slices using [QuickNII](https://quicknii.readthedocs.io/en/latest/), followed by [VisuAlign](https://visualign.readthedocs.io/en/latest/) software. Save the json file from VisuAlign for use in next steps.

**Get LCM sample coordinates in Allen CCF**
Registered input from VisuAlign is in the form of json file with o, u, v 3d vectors and markers, which providing positional and non-linear transformation information. These data are used to transform each slice into allen ccf coordinates and extract coordinates, extended to 3d volumes from one slice to the next. Functions in LCM_registration_functions.py and visualign_functions.py in lcm_registration do this for you.

N.B. Make sure to update lcm registration parameters in parameters yaml file in Sequencing folder.

**lcm_registration function pipeline:**
```
convert_images
    |
    v
get_euclidean_distance -> calc_euclidean_distance
    |
    v       
group_ROI_coordinates
    |
    v
generate_region_table_across_samples -> get_acronymn
    |
    v
   END
```
**Pre-process sequencing data**
Make sure to update the parameters.yml file in preprocessing_sequencing folder. A copy is then saved into the project folder, and you then update that one from then on.
All the functions for preprocessing are in the preprocess_sequences.py script. Initially run 'start_splitting.py' in the '/scripts' folder, and each of the functions will run sequentially up to the point where you need to manually adjust parameters for UMI count cut-off and template switching cut-off. At this point, run the notebook in '/preprocessing_sequencing/notebooks/determine_UMI_cutoff_and_template_switching_thresholds.ipynb' to visualise UMI count distribution etc., adjust parameters, and run the next section to finish preprocessing.

## Installation

Should be easy enough. However the environment *must* be called `MAPseq_processing`.
Clone the repo and install with pip. This requires to have access to the lab github 
repositories.
I could not install `umi_tools` via pip (maybe because the python version was too
recent), so I let conda deal with that

```
conda create -n MAPseq_processing -c bioconda -c conda-forge umi_tools pip
conda activate MAPseq_processing
pip install -e .
```