import os
import numpy as np
import pandas as pd
from umi_tools import UMIClusterer
from datetime import datetime
import subprocess, shlex
import pathlib


# separately batch process UMI and barcodes
def process_umi_and_barcode(barcode_range):
    """Function to generate batch jobs for all samples in the dataset, processing neuron barcodes and UMI's separately, before they are joined together later.
    Args:
        barcode_range (int): number of barcodes in list

    Returns:
        None.

    """
    batch_script = "/camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Bash_scripts/batch_preprocess.sh"
    for x in range(barcode_range):
        barcode_num = "BC" + str(x + 1) + ".csv"
        command_neuron = f"sbatch {batch_script} {barcode_num} neuron"
        print(command_neuron)
        subprocess.Popen(shlex.split(command_neuron))
        command_umi = f"sbatch {batch_script} {barcode_num} umi"
        print(command_umi)
        subprocess.Popen(shlex.split(command_umi))


def preprocess_reads(barcode, directory, sequence_type, homopolymer_thresh=5):
    """Function to run UMI-tools to correct PCR and sequencing errors for UMI's and neuron barcodes separately, then take UMI counts for each barcode.

    Args:
        barcode (str): which sample barcode file to look at
        directory (str): directory where sample splitting occurred
        sequence_type (str): either 'neuron' or 'umi' -> changes which part of sequence to be analysed
        homopolymer_thresh (int): we use a cut-off for homopolymeric repeats (a common sequencing error). default=5

    Returns:
        None.
    """
    directory_path = pathlib.Path(directory)
    corrected_path = directory_path.joinpath("preprocessed_seq")
    pathlib.Path(corrected_path).mkdir(parents=True, exist_ok=True)
    big_output_directory_path = directory_path.joinpath("temp_big_ones")
    pathlib.Path(big_output_directory_path).mkdir(parents=True, exist_ok=True)
    hopolA = "A" * homopolymer_thresh
    hopolT = "T" * homopolymer_thresh
    hopolC = "C" * homopolymer_thresh
    hopolG = "G" * homopolymer_thresh
    barcode_file = directory_path.joinpath(barcode)
    print(
        f"Starting {sequence_type} collapsing for {barcode_file.stem} at {datetime.now()}"
    )
    raw_bc = pd.read_csv(
        barcode_file, delimiter="\t", skiprows=lambda x: (x != 0) and not x % 2
    )
    barcode_tab = pd.DataFrame()
    barcode_tab["full_read"] = raw_bc[
        (~raw_bc.iloc[:, 0].str.contains("N"))
        & (~raw_bc.iloc[:, 0].str.contains(hopolA))
        & (~raw_bc.iloc[:, 0].str.contains(hopolG))
        & (~raw_bc.iloc[:, 0].str.contains(hopolC))
        & (~raw_bc.iloc[:, 0].str.contains(hopolT))
    ]
    if sequence_type == "neuron":
        barcode_tab["sequence"] = barcode_tab["full_read"].str[:32]
    elif sequence_type == "umi":
        barcode_tab["sequence"] = barcode_tab["full_read"].str[32:46]
    if (
        len(barcode_tab) > 10000000
    ):  # if the size of the barcode table is huge, we'll split it into smaller chunks for processing, then cluster together again at end
        numberdivisions = int((len(barcode_tab) / 1000000) // 1)
        sequences = barcode_tab
        for i in range(numberdivisions):
            if i != numberdivisions - 1:
                df_short = sequences.iloc[:1000000, :]
                sequences = sequences.iloc[1000000:, :]
            else:
                df_short = sequences
            newtmpfile = big_output_directory_path.joinpath(
                f"{sequence_type}_intermediate_{barcode_file.stem}_{i + 1}.csv"
            )
            df_short.to_csv(newtmpfile)
        batch_collapse_barcodes(
            newdir=big_output_directory_path,
            sequence_type=sequence_type,
            barcodenum=barcode_file.stem,
            numberdivisions=numberdivisions,
        )
        print(f"split {barcode_file.stem} into {numberdivisions} repetitions")
    else:
        corrected_sequencess = error_correct_sequence(reads=barcode_tab)
        new_file = corrected_path.joinpath(
            f"{sequence_type}_corrected_{barcode_file.stem}.csv"
        )
        corrected_sequencess.to_csv(new_file)
        print(f"Finished at {datetime.now()}")


def error_correct_sequence(reads):
    """Function to perform error correction using UMI tools directional adjacency approach
    Args:
        reads: table that has the reads that your interested in (has to be with specific column headers seen in preprocess_reads function)
    Returns: table containing corrected umi's
    """
    tstart = datetime.now()
    barcode_tab = reads
    clusterer = UMIClusterer(cluster_method="directional")
    barcode_tab["byt_sequence"] = [x.encode() for x in barcode_tab.sequence]
    umi_counts = barcode_tab["byt_sequence"].value_counts()
    umi_dict = umi_counts.to_dict()
    clustered_umis = clusterer(umi_dict, threshold=1)
    my_dict = {}
    for x in clustered_umis:
        for each in x:
            my_dict[each] = x[0]
    barcode_tab["corrected_sequences"] = [
        my_dict[x].decode() for x in barcode_tab.byt_sequence
    ]
    corrected = sum(barcode_tab.sequence != barcode_tab.corrected_sequences)
    total = len(barcode_tab)
    corrected_sequences = barcode_tab["corrected_sequences"]
    print(f"Corrected {corrected} out of  {total} neuron barcode counts")
    tend = datetime.now()
    print(f"That took {tend - tstart}", flush=True)
    return corrected_sequences


def batch_collapse_barcodes(newdir, sequence_type, barcodenum, numberdivisions):
    """
    Function to send a load of slurm jobs for bigger files to process in chunks of 5mil reads, so it won't take a lifetime to process.

    Args:
        newdir: where to big intermediate files are kept
        barcodenum: RT sample barcode identifier
        numberdivisions: the number of chunks you have for each bigger file
    Returns:
        None.
    """
    big_directory = pathlib.Path(newdir)
    script_path = pathlib.Path(
        "/camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Bash_scripts/process_chunks.sh"
    )
    for x in range(numberdivisions):
        to_read = big_directory.joinpath(
            f"{sequence_type}_intermediate_{barcodenum}_{x + 1}.csv"
        )
        command = f"sbatch {script_path} {to_read}"
        print(command)
        subprocess.Popen(shlex.split(command))


def process_chunks(to_read, directory):
    """Function to process the chunks of 1mil reads from bigger files, to correct sequencing errors, so it won't take a lifetime to process.
    Args:
        to_read: split barcode csv file to read
        directory: where files are
    Returns:
        None.
    """
    big_directory_path = pathlib.Path(directory)
    barcode_path = big_directory_path.joinpath(to_read)
    barcode_tab = pd.read_csv(barcode_path)
    print(f"Starting UMI collapsing for {barcode_path.stem} at {datetime.now()}")
    corrected_sequencess = error_correct_sequence(reads=barcode_tab)
    new_file = big_directory_path.joinpath(f"corrected_{barcode_path.stem}.csv")
    corrected_sequencess.to_csv(new_file)
    print(f"Finished at {datetime.now()}")
