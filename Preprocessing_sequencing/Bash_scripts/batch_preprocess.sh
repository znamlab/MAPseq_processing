#!/bin/sh
# Job name
#SBATCH --job-name=batch_process_210323
# Number of tasks in job script
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=250G

# Notifications
#SBATCH --output=/camp/home/turnerb/slurm_logs/March23/batchprocess_$1_%j.out
#SBATCH --error=/camp/home/turnerb/slurm_logs/March23/batchprocess_$1_%j.err
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
python Call_functions_batch.py $1 $2
echo "Done"
