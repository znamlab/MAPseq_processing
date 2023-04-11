#!/bin/sh
# Job name
#SBATCH --job-name=process_big_one
# Number of tasks in job script
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=250G


# Notifications
#SBATCH --output=/camp/home/turnerb/slurm_logs/April23/100423calling_big_one_%j.out
#SBATCH --error=/camp/home/turnerb/slurm_logs/April23/100423calling_big_one_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benita.turner-bridger@crick.ac.uk
# Main script that calls all the steps of the preprocessing

# load relevant modules
echo "Loading modules"
ml purge

ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Activating environment"
conda activate MAPseq_processing

echo "Running MAPseq preprocessing for $1"


cd /camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Preprocessing_scripts
python call_big_one_processing.py $1
echo "Done"
