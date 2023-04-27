#!/bin/sh
# Job name
#SBATCH --job-name=template_switch_check
# Number of tasks in job script
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=350G
#SBATCH --partition=hmem

# Notifications
#SBATCH --output=/camp/home/turnerb/slurm_logs/April23/2404_template_switch_check_%j.out
#SBATCH --error=/camp/home/turnerb/slurm_logs/April23/2404_template_switch_check_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benita.turner-bridger@crick.ac.uk
# Main script that calls all the steps of the preprocessing

# load relevant modules
echo "Loading modules"
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

echo "Activating environment"
conda activate MAPseq_processing

cd /camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Preprocessing_scripts
python call_template_switching_check.py
echo "Done"
