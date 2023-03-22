#!/bin/sh
# Job name
#SBATCH --job-name=210323_sample_splitting

# Number of tasks in job script
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=350G
#SBATCH --partition=hmem

# Notifications
#SBATCH --output=/camp/home/turnerb/slurm_logs/March23/210323_splitting_preprocess_%j.out
#SBATCH --error=/camp/home/turnerb/slurm_logs/March23/210323_splitting_preprocess_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benita.turner-bridger@crick.ac.uk
# Main script that calls all the steps of the preprocessing

# load relevant modules
echo "Loading modules"
ml Anaconda3
ml FASTX-Toolkit
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Activating environment"
conda activate MAPseq_processing

echo "Running MAPseq preprocessing"

cd /camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Preprocessing_scripts
python Call_functions_splitting.py



echo "Done"
