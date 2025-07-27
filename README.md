# MAPseq_processing
Pipeline for processing MAPseq NGS datasets

**Image naming conventions:**
Images of each LCM section are taken on the LCM microscope and aligned to same images used for registration so that LCM regions can be mapped. An overview section of each image must be taken before performing LCM. You must take the image on the side of the section that is most anterior and name each section to end with: "_s[number of section]". Numbering of sections goes from anterior to posterior - e.g. "_s001" for OB, even if you didn't slice this section first. For each lcm image, you must name the image by section first, then tube name e.g. "s001_tube1".

**Generate ROI masks for each LCM sample:**
First update LCM parameters yaml file with directory name, mouse name, project name etc.
Using matlab, run the lcm_registration/Align_LCM_to_section.m matlab script. LCM images are looped through per slice (you need to specify which section you want to do in the script, as well as the directory) and you need to manually draw a polygon ROI in the images that pop up - in red will be highlighted the difference between the image before and after LCM, so you can easily identify where the ROI is. If you accidentally draw one ROI wrong, use lcm_registration/Align_LCM_to_single_section.m so you don't want to loop though all the tubes again. If there are errors in registration due to deformed image/bad stitching go to either Correctingbadstitching.m (where you need to select the badly stitched frames within the image and select a point that is continuous on the slice, do this iteratively if you want multiple frames to be registered, then repeat the Align_LCM_to_section script with this corrected image as input). For Point_alignment.m you just need to select the same regions in each image so that it is manually aligned.

**Register sections to Allen atlas.**
Register slices using [QuickNII](https://quicknii.readthedocs.io/en/latest/), followed by [VisuAlign](https://visualign.readthedocs.io/en/latest/) software. Save the json file from VisuAlign for use in next steps.

**Get LCM sample coordinates in Allen CCF.**
Implemented by functions in LCM_registration_functions.py and visualign_functions.py in lcm_registration. The registered output from QuickNII and VisuAlign is in the form of a json file containing o, u, v 3D vector components and markers that provide positional and non-linear transformation information. These data are then used to transform each pixel in each coronal slice into Allen CCF coordinates. 2D LCM slices are extended to 3D volumes from one slice to the next by taking the z coordinates for each pixel in each slice and finding the z coordinates in the preceding slice for pixels that had the closest Euclidean distance in x,y coordinates. The thickness of the coronal slice is then determined as the mean difference in z coordinate values between image pixels with the closest 20% of Euclidean distances. To calculate the Allen CCF coordinates for each LCM cubelet, we used the binary ROI mask (generated as output from the MATLAB scripts above) to extract Allen CCF coordinates from each 3D coronal slice and annotate every coordinate in the LCM sample to a 3D numpy array in the dimensions of the 25um resolution Allen brain atlas.

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
**Pre-process sequencing data.**
Make sure to update the parameters.yml file in preprocessing_sequencing folder. A copy is then saved into the project folder, and you then update that one from then on.
All the functions for preprocessing are in the preprocess_sequences.py script. Initially run 'start_splitting.py' in the '/scripts' folder, and each of the functions will run sequentially up to the point where you need to manually adjust parameters for UMI count cut-off and template switching cut-off. At this point, run the notebook in '/preprocessing_sequencing/notebooks/determine_UMI_cutoff_and_template_switching_thresholds.ipynb' to visualise UMI count distribution etc., adjust parameters, and run the next section to finish preprocessing.

**preprocess_sequences function pipeline:**
```
ps.split_samples
    |
    v
preprocess_reads  --> process_neuron_barcodes
    |
    v
correct_all_umis (dep: process_neuron_barcodes) -> correct_umi_sequences
    |
    v       
collate_error_correction_results (dep: correct_all_umis)
    |
    v
join_tabs_and_split -> switch_analysis
    |
    v
   END
```        

**Final processing of MAPseq datasets.**
The output of preprocess_sequences gives a series of .csv files corresponding to barcodes in each sample. To create a dataframe with barcode counts across samples for each barcode, perform various QCs, and selection of only neurons with soma in particular brain areas (in our case A1), run  `barcode_match_qc_final_process.ipynb` in final_processing for each mouse. Make sure you have adjusted the "general_analysis_parameters.yaml" in the final_processing folder to contain your specific file paths. To generate shuffle population with different area assignment approaches, run `generate_shuffle_pop.ipynb`.


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