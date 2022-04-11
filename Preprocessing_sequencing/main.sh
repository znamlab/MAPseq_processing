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

echo "Running MAPseq preprocessing"
python Call_functions.py



echo "Done"
