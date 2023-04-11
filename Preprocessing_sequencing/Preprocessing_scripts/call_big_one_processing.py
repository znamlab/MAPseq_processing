import subprocess
import os
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np
from umi_tools import UMIClusterer

# from get_corr_UMI import getCorrectUMIs
import sys
import shlex
from preprocess_sequences import process_barcode_tables
from preprocess_sequences import preprocess_reads
from preprocess_sequences import error_correct_sequence


camp_root = pathlib.Path("/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq")
raw = camp_root / "A1_MAPseq/FIAA32.6a/Sequencing/Raw_data"
out_dir = camp_root / "A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/UpdatedApril"
acqid = "TUR5514A2"
barcodes = camp_root / "Sequencing/Reference_files/sample_barcodes.txt"
barcode_file = str(sys.argv[1])


process_barcode_tables(
    barcode=barcode_file, directory=out_dir, homopolymer_thresh=5, big_mem="yes"
)
