from preprocessing_sequencing import preprocess_sequences as ps
from datetime import datetime

parameters = ps.load_parameters(directory="root")
print("parameters loaded", flush=True)
# start job for splliting samples
job_id = ps.split_samples(use_slurm=True, slurm_folder=parameters["SLURM_DIR"])
t = datetime.now()
print(f"Sent first job at {t}", flush=True)
#
