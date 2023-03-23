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

number_samples = int(sys.argv[1])

process_umi_and_barcode(barcode_range=number_samples)
