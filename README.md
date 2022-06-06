# MAPseq_processing
Pipeline for processing MAPseq NGS datasets

Images of each LCM section are taken on the LCM microscope and aligned to same images used for registration so that LCM regions can be mapped.

Here, use LCM_registration/Align_LCM_to_section.m matlab script. LCM images are looped through per slice and you need to manually draw ROI.

If there are errors in registration due to deformed image/bad stitching go to either Correctingbadstitching.m (where you need to select the badly stitched frames within the image and select a point that is continuous on the slice, do this iteratively if you want multiple frames, then repeat the aligning script). For Point_alignment.m you just need to select the same regions in each image so that it is manually aligned.

Subsequently, data from registered slices (here, using QuickNI and VisuAlign software) is used for connecting ROI to brain region using the script (* still WIP, Slice_to_allen_CCF). Here, registered input is in the form of json file with o, u, v 3d vectors and markers providing positional and non-linear transformation information. These data are used to transform each slice into allen ccf coordinates and extract coordinates, extended to 3d volumes from one slice to the next.

For pre-processing MAPseq datasets, use the main.sh script to call individual python scripts that each do a different job (specified in the name e.g. aligning UMI's) and should be called in the following order:
(1) sample_splitting.py
(2) UMI_grouping.py
(3) collapse_UMIs.py
(4) align_barcodes.py
(5) barcode_collapsing.py

Sample barcode sequences are specified in pre-processing as a txt file for reference.
