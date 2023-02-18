#!/bin/sh
# Job name
#SBATCH --job-name=takethesmallones_UMI
# Number of tasks in job script
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=50G

# Notifications
#SBATCH --output=/camp/home/turnerb/slurm_logs/BCBatch/180223/takethesmallones_UMI_%j.out
#SBATCH --error=/camp/home/turnerb/slurm_logs/BCBatch/180223/takethesmallones_UMI_%j.err
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

echo "Running MAPseq preprocessing"

#BARCODE=$(</camp/home/turnerb/home/shared/code/MAPseq_processing/AC_MAPseq/Brain1_FIAA32.6a/New_with_UMItools/160223/temp/temp_sampleUMItotakeFiles.txt)

cd /camp/home/turnerb/home/shared/code/MAPseq_processing/AC_MAPseq/Brain1_FIAA32.6a/New_with_UMItools/160223/files/UMIbatch
python Call_functions_UMIchunk.py $1
echo "Done"
