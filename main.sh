#!/bin/sh

# Main script that calls all the steps of the preprocessing

# load relevant modules
echo "Loading modules"
ml Anaconda3
ml FASTX-Toolkit
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Activating environement"
conda activate flexenv

echo "Splitting samples"
python 01_sample_splitting.py

echo "Done"
