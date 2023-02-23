import subprocess
import os
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umi_tools import UMIClusterer
#from get_corr_UMI import getCorrectUMIs
from get_corr_neuronBC import getCorrectBarcodes
import sys
from get_corr_neuronBC import processtheminlots
import shlex
from get_corr_neuronBC import processthechunks
from get_corr_neuronBC import combine_the_chunks
from get_corr_neuronBC import combineUMIandBC
from get_corr_neuronBC import normalise_on_spike
from barcode_matching import barcode_matching

#run sample Splitting
camp_root = pathlib.Path('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq')
raw = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Raw_data/'
out_dir = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split'
#out_dir = '/camp/lab/znamenskiyp/home/shared/code/MAPseq_processing/AC_MAPseq/Brain1_FIAA32.6a/New_with_UMItools'
acqid = 'TUR5514A2'
barcodes = camp_root / 'Sequencing/Reference_files/sample_barcodes.txt'
sorting_directory = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/sorting'
#barcodefile = sys.argv[1]
bigonesorting_dir = camp_root / 'A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/bigones'
#barcodenum= str(sys.argv[1])
#getCorrectUMIs(barcodefile= barcodefile, directory=out_dir, homopolymerthresh = 5)
#processtheminlots(directory=bigonesorting_dir, barcodenum='BC1', numberdivisions=2)

#combine_the_chunks(directory='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/bigones', outdirectory='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/noones', barcodefile='BC35')
#combine_the_chunks(directory='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/bigones', outdirectory='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/noones', barcodefile='BC57')
#combine_the_chunks(directory='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/bigones', outdirectory='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/noones', barcodefile='BC64')
#combineUMIandBC(directory='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/', outdirectory ='/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/BC_split/temp/noones', barcodefilerange=96)
directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/final_counts/noones'
#normalise_on_spike(directory=directory, outdir=directory)
barcode_matching(sorting_directory=directory, num_samples=91)