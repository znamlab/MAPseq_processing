#!/bin/sh
# Job name
#SBATCH --job-name=180223_UMIcollapsewithUMItools
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=50G


# Notifications
#SBATCH --output=/camp/home/turnerb/slurm_logs/BCBatch/180223/180223_UMIcollapsewithUMItools_%j.out
#SBATCH --error=/camp/home/turnerb/slurm_logs/BCBatch/180223/180223_UMIcollapsewithUMItools_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benita.turner-bridger@crick.ac.uk
#SBATCH --array=1-96
# Main script that sends out individuals jobs for barcode clustering the steps of the  MAPseq preprocessing

# Specify the path to the config file
config=/camp/home/turnerb/home/shared/code/MAPseq_processing/AC_MAPseq/Brain1_FIAA32.6a/New_with_UMItools/160223/files/BarcodeFiles.txt

# Extract the sample name focat r the current $SLURM_ARRAY_TASK_ID
barcodefile="$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)"

# load relevant modules
echo "Loading modules"
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Activating environment"
conda activate MAPseq_processing

echo "Running MAPseq preprocessing UMI correcting"
python Call_functions_UMI.py ${barcodefile}
echo "This is array task ${SLURM_ARRAY_TASK_ID}, the sample name is ${barcodefile}"