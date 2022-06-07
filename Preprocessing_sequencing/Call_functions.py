import shutil
import subprocess
import os
import pathlib
from datetime import datetime
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import (draw, DiGraph, Graph)

from sample_splitting import split_samples
from sample_splitting import run_bc_splitter
from sample_splitting import unzip_fastq

from UMI_grouping import combineL00
from UMI_grouping import groupingUMI_restructure
from UMI_grouping import UMI_bowtie
from UMI_grouping import restructurebowtie

from collapse_UMIs import sortUMIs

from align_barcodes import collapsedUMIbarcode
from align_barcodes import bowtiebarcodes

from barcode_collapsing import get_int_bc
from barcode_collapsing import barcodecollapsing

#run sample Splitting
camp_root = pathlib.Path('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq')
raw = camp_root / 'Sequencing/Raw_data/BRAC5676.1h/140422_full_run/210422'
out_dir = camp_root / 'Sequencing/Processed_data/BRAC5676.1h/140422_full_run/230422'
acqid = 'TUR4752A1'
barcodes = camp_root / 'Sequencing/Reference_files/sample_barcodes.txt'
sorting_directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/140422_full_run/230422/sorting'

split_samples(raw_dir=raw, output_dir=out_dir, acq_id=acqid, barcode_file=barcodes,
                  n_mismatch=1, r1_part=None, r2_part=(0,30), verbose=1)
groupingUMI_restructure(directory=out_dir)
UMI_bowtie(directory=out_dir)
restructurebowtie(directory=out_dir)
sortUMIs(directory=out_dir, minUMI=1)
collapsedUMIbarcode(directory=out_dir)
bowtiebarcodes(directory=out_dir)
barcodecollapsing(directory=out_dir, minbarcode=0)
barcode_matching(sorting_directory= sorting_directory, num_samples=96)
