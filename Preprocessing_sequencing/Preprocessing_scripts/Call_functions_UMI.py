import os
import subprocess
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umi_tools import UMIClusterer
from get_corr_UMI import UMIprocessthechunks
from get_corr_UMI import UMIprocesstheminlots
from get_corr_UMI import getCorrectUMIs
import sys
import shlex

#run sample Splitting
camp_root = pathlib.Path('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq')
raw = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Raw_data/'
out_dir = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split'
#out_dir = '/camp/lab/znamenskiyp/home/shared/code/MAPseq_processing/AC_MAPseq/Brain1_FIAA32.6a/New_with_UMItools'
acqid = 'TUR5514A2'
barcodes = camp_root / 'Sequencing/Reference_files/sample_barcodes.txt'
sorting_directory = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/sorting'
barcodefile = str(sys.argv[1])
bigonesorting_dir = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/bigones'

getCorrectUMIs(barcodefile= barcodefile, directory=out_dir, bigonesorting_dir=bigonesorting_dir, homopolymerthresh = 5)

