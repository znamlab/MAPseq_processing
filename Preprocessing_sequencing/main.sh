#!/bin/sh

# Main script that calls all the steps of the preprocessing

# load relevant modules
echo "Loading modules"
ml Anaconda3
ml FASTX-Toolkit
ml Bowtie
source /camp/apps/eb/software/Anaconda/conda.env.sh

#need to make a camp env
#echo "Activating environment"
#conda activate flexenv

echo "Splitting samples"
python 01_sample_splitting.py

echo "Aligining UMIs to eachother"
python 02_UMI_grouping.py

echo "Collapsing UMIs"
python 03_collapse_UMIs.py

echo "Aliging barcodes to eachother"
python 04_align_barcodes.py

echo "Collapsing the barcodes"
python 05_barcode_collapsing.py

echo "Done"
