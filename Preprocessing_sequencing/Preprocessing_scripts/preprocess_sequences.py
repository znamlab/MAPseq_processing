import os
import numpy as np
import pandas as pd
from umi_tools import UMIClusterer
from datetime import datetime
import subprocess, shlex
import pathlib


def preprocess_reads(directory, barcode_range, homopolymer_thresh=5):
    """Function to run UMI-tools to correct PCR and sequencing errors for UMI's and neuron barcodes separately, then take UMI counts for each barcode.

    Args:
        directory (str): directory where sample splitting occurred
        barcode_range (int): number of barcodes in list
        homopolymer_thresh (int): we use a cut-off for homopolymeric repeats (a common sequencing error). default=5

    Returns:
        None.
    """
    # for loop to error correct neuron barcodes and umi's sequentially, but if file is large, generate a separate script to process
    # barcodes with higher memory, then take each neuron barcode and correct umi's within each
    directory_path = pathlib.Path(directory)
    big_one_script = pathlib.Path(
        "/camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Bash_scripts/process_big_ones.sh"
    )
    for x in range(barcode_range):
        barcode_num = "BC" + str(x + 1) + ".txt"
        barcode_file = directory_path.joinpath(barcode_num)
        print(f"Starting collapsing for {barcode_file.stem} at {datetime.now()}")
        raw_bc = pd.read_csv(
            barcode_file, delimiter="\t", skiprows=lambda x: (x != 0) and not x % 2
        )
        if len(raw_bc) > 10000000:
            # if the size of the barcode table is huge, we'll process the less diverse neuron barcodes first with bigger memory batch script,
            # then we'll process UMI's a few barcodes at a time
            command = f"sbatch {big_one_script} {barcode_file}"
            print(command)
            subprocess.Popen(shlex.split(command))
        else:
            process_barcode_tables(
                barcode=barcode_file,
                directory=directory,
                homopolymer_thresh=homopolymer_thresh,
                big_mem="no",
            )


def process_barcode_tables(barcode, directory, homopolymer_thresh, big_mem):
    """Function to process the UMI and barcode sequences, removing homopolymers before calling error correction. If the sample barcode has a large number
    of reads, we'll send another batch script to handle these separately.
    Args:
        barcode: what RT barcode file for samples is being analysed
        directory: directory where sample barcode files are kept
        homopolymer_thresh: number of homopolymeric repeats, default =5
        big_mem: 'yes' or 'no'. if the txt file is a big one or not.
    """
    directory_path = pathlib.Path(directory)
    barcode_file = pathlib.Path(barcode)
    corrected_path = directory_path.joinpath("preprocessed_seq")
    pathlib.Path(corrected_path).mkdir(parents=True, exist_ok=True)
    big_output_directory_path = directory_path.joinpath("temp_big_ones")
    pathlib.Path(big_output_directory_path).mkdir(parents=True, exist_ok=True)
    hopolA = "A" * homopolymer_thresh
    hopolT = "T" * homopolymer_thresh
    hopolC = "C" * homopolymer_thresh
    hopolG = "G" * homopolymer_thresh
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
    barcode_tab["neuron_sequence"] = barcode_tab["full_read"].str[:32]
    barcode_tab["umi_sequence"] = barcode_tab["full_read"].str[32:46]
    # error correct neuron barcodes first (not as diverse as umi's, so whole table processed for big ones)
    neuron_bc_corrected = error_correct_sequence(
        reads=barcode_tab, sequence_type="neuron"
    )
    if big_mem == "no":
        corrected_sequences = error_correct_sequence(
            reads=neuron_bc_corrected, sequence_type="umi"
        )
    elif big_mem == "yes":
        corrected_sequences = neuron_bc_corrected
        neuron_list = corrected_sequences["corrected_sequences_neuron"].unique()
        corrected_sequences["corrected_sequences_umi"] = "NA"
        n = 10000  # number of neuron barcodes to process at one time
        neuron_list_subsets = [
            neuron_list[i : i + n] for i in range(0, len(neuron_list), n)
        ]
        for sequence_str in neuron_list_subsets:
            neuron_bc_analysed = corrected_sequences[
                corrected_sequences["corrected_sequences_neuron"].isin(sequence_str)
            ]
            chunk_analysed = error_correct_sequence(
                reads=neuron_bc_analysed, sequence_type="umi"
            )
            corrected_sequences.update(chunk_analysed)
    new_file = corrected_path.joinpath(f"corrected_{barcode_file.stem}.csv")
    corrected_sequences.to_csv(new_file)
    print(f"Finished at {datetime.now()}")


def error_correct_sequence(reads, sequence_type):
    """Function to perform error correction using UMI tools directional adjacency approach
    Args:
        reads: table that has the reads that your interested in (has to be with specific column headers seen in preprocess_reads function)
        sequence_type (str): 'neuron' or 'umi'. Are the sequences neuron barcodes or umi's?
    Returns: table containing corrected sequences
    """
    tstart = datetime.now()
    barcode_tab = reads
    clusterer = UMIClusterer(cluster_method="directional")
    if sequence_type == "neuron":
        sequence = "neuron_sequence"
    elif sequence_type == "umi":
        sequence = "umi_sequence"
    barcode_tab["byt_sequence"] = [x.encode() for x in barcode_tab[f"{sequence}"]]
    sequence_counts = barcode_tab["byt_sequence"].value_counts()
    sequence_dict = sequence_counts.to_dict()
    clustered_sequences = clusterer(sequence_dict, threshold=1)
    my_dict = {}
    for x in clustered_sequences:
        for each in x:
            my_dict[each] = x[0]
    barcode_tab[f"corrected_sequences_{sequence_type}"] = [
        my_dict[x].decode() for x in barcode_tab.byt_sequence
    ]
    corrected = sum(
        barcode_tab[f"{sequence}"]
        != barcode_tab[f"corrected_sequences_{sequence_type}"]
    )
    total = len(barcode_tab)
    print(f"Corrected {corrected} out of  {total} sequence counts")
    tend = datetime.now()
    print(f"That took {tend - tstart}", flush=True)
    return barcode_tab


def combineUMIandBC(directory, UMI_cutoff=7, barcode_filerange=96):
    """
    Function to combine corrected barcodes and UMI's for each read and collect value counts.
    Also to detect degree of template switching between reads by seeing if UMI is shared by more than one barcode
    Also to split spike RNA from neuron barcodes, by whether contains N[24]ATCAGTCA (vs N[32]CTCT for neuron barcodes)
    Args:
        directory (str): path to temp file where the intermediate UMI and barcode clustered csv files are kept
        UMI_cutoff (int): threshold for minimum umi count per barcode. (default =7)
        barcode_filerange (int): the number of samples you want to loop through (default set for 96)
    """
    dir_path = pathlib.Path(directory)
    out_dir = dir_path.joinpath("Final_processed_sequences")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(barcode_filerange):
        os.chdir(directory)
        barcode = f"BC{i+1}"
        sample_file = dir_path / f"corrected_{barcode}.csv"
        if os.path.isfile(sample_file):
            print("processing %s" % barcode, flush=True)
            sample_table = pd.read_csv(sample_file)
            sample_table["combined"] = (
                sample_table["corrected_sequences_neuron"]
                + sample_table["corrected_sequences_umi"]
            )
            spike_in = sample_table[
                sample_table["combined"].str.contains("^.{24}ATCAGTCA") == True
            ].rename_axis("sequence")
            neurons = sample_table[
                sample_table["combined"].str.contains("^.{30}[CT][CT]") == True
            ].rename_axis("sequence")
            neuron_counts = (
                neurons["combined"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            spike_counts = (
                spike_in["combined"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            # only take umi counts greater or equal to minimum UMI cutoff
            neuron_counts = neuron_counts[neuron_counts["counts"] >= UMI_cutoff]
            spike_counts = spike_counts[spike_counts["counts"] >= UMI_cutoff]
            spike_counts["barcode"] = spike_counts["sequence"].str[:32]
            neuron_counts["barcode"] = neuron_counts["sequence"].str[:32]
            spike_counts = (
                spike_counts["barcode"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            neuron_counts = (
                neuron_counts["barcode"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            os.chdir(out_directory)
            print("finished %s" % barcode, flush=True)
            to_save_BC = out_dir / f"neuron_counts_{barcode}.csv"
            to_save_spike = out_dir / f"spikecounts_{barcode}.csv"
            spike_counts.to_csv(to_save_spike)
            neuron_counts.to_csv(to_save_BC)
        else:
            print(f"file not there for {barcode}")


def barcode_matching(sorting_directory, num_samples):
    """
    Function to identify matching barcodes between samples
    Args: sorting directory =sorting directory where bowtieoutput files are saved
                num_samples = number of sample barcodes used
    """
    sorting_dir = pathlib.Path(sorting_directory)
    all_seq = []
    for barcode_file in os.listdir(sorting_dir):
        if barcode_file.startswith("neuron_counts_"):
            print(f"reading barcode file{barcode_file}", flush=True)
            to_read = pd.read_csv(barcode_file)
            sequences = to_read["sequence"]
            all_seq.append(sequences)

    all_seq = pd.DataFrame(all_seq)
    bla = all_seq.to_numpy().flatten()
    all_seq = [x for x in bla if str(x) != "nan"]
    all_seq_unique = np.unique(all_seq)

    # tabulate counts of each barcode in each sample area

    samples = list(range(1, num_samples + 1))
    barcodes_across_sample = pd.DataFrame(columns=samples, dtype=int)

    index = -1
    for barcode in all_seq_unique:
        index += 1
        for barcode_file in os.listdir(sorting_directory):
            if barcode_file.startswith("neuron_counts_"):
                to_read = pd.read_csv(barcode_file)
                sample = int(
                    barcode_file.split("neuron_counts_BC", 1)[1][: -len(".csv")]
                )
                for r, sequence in to_read["sequence"].items():
                    if sequence == barcode:
                        barcodes_across_sample.at[index, sample] = to_read["counts"][r]
    print("finito")
    barcodes_across_sample.to_pickle(sorting_dir / "barcodes_across_sample.pkl")


def normalise_on_spike(directory, outdir):
    """
    Function to normalise differences in RT efficiency and 2nd strand synth between samples by
    normalising barcode counts for each sample based on counts for spike-in RNA

    Args:
        directory: Directory where counts are kept
        outdir: directory where output is saved
    """
    # first make a table with total spike counts per sample
    # take min spike count as 1, and normalise counts to this
    os.chdir(directory)
    spike_counts = pd.DataFrame(columns=["sample", "spike_count"])
    for sample in os.listdir(directory):
        if sample.startswith("spikecounts"):
            samplename = sample.split("spikecounts_", 1)
            samplename = samplename[1][: -len(".csv")]
            sample1 = pd.read_csv(sample)
            sample1["counts"] = sample1["counts"].astype("int")
            sumcounts = sample1["counts"].sum()
            new_row = pd.DataFrame(
                {"sample": samplename, "spike_count": sumcounts}, index=[0]
            )
            spike_counts = pd.concat([spike_counts, new_row])
    lowest = min(
        spike_counts["spike_count"]
    )  # NB ... because max spike number is not always 1, might need to adjust umi count/cutoff
    spike_counts["normalisation_factor"] = spike_counts["spike_count"] / lowest
    print(spike_counts)
    for sample in os.listdir(directory):
        os.chdir(directory)
        if sample.startswith("neuroncounts_"):
            samplename = sample.split("neuroncounts_", 1)
            samplename = samplename[1][: -len(".csv")]
            sample1 = pd.read_csv(sample)
            normalisation_factor = spike_counts.loc[
                spike_counts["sample"] == samplename, "normalisation_factor"
            ].iloc[0]
            sample1["normalised_counts"] = sample1["counts"] / normalisation_factor
            name = "normalised_counts_%s.csv" % samplename
            os.chdir(outdir)
            sample1.to_csv(name)
