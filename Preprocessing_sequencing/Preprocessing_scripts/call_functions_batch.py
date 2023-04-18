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
from preprocess_sequences import process_umi_and_barcode
from preprocess_sequences import preprocess_reads
from preprocess_sequences import error_correct_sequence
from preprocess_sequences import batch_collapse_barcodes


camp_root = pathlib.Path("/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq")
raw = camp_root / "A1_MAPseq/FIAA32.6a/Sequencing/Raw_data"
out_dir = camp_root / "A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/New"
acqid = "TUR5514A2"
barcodes = camp_root / "Sequencing/Reference_files/sample_barcodes.txt"
barcodenum = str(sys.argv[1])
sequence_type = str(sys.argv[2])

# getCorrectUMIs(barcodefile= barcodefile, directory=out_dir, homopolymerthresh = 5)
# processtheminlots(directory=bigonesorting_dir, barcodenum='BC1', numberdivisions=2)

preprocess_reads(
    barcode=barcodenum,
    directory=out_dir,
    sequence_type=sequence_type,
    homopolymer_thresh=5,
)
