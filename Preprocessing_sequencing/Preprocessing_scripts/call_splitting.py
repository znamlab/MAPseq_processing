import shutil
import subprocess
import os
import pathlib
from datetime import datetime
import gzip
from sample_splitting import split_samples
from sample_splitting import run_bc_splitter
from sample_splitting import unzip_fastq


# run sample Splitting
camp_root = pathlib.Path("/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq")
raw = camp_root / "A1_MAPseq/FIAA32.6a/Sequencing/Raw_data"
out_dir = camp_root / "A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/New"
acqid = "TUR5514A2"
barcodes = camp_root / "Sequencing/Reference_files/sample_barcodes.txt"

split_samples(
    raw_dir=raw,
    output_dir=out_dir,
    acq_id=acqid,
    barcode_file=barcodes,
    n_mismatch=2,
    r1_part=(0, 32),
    r2_part=(0, 30),
    verbose=1,
)
