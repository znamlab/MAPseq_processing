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
                big_mem="no",
            )


def process_barcode_tables(barcode, directory, big_mem):
    """Function to process the UMI and barcode sequences, removing homopolymers before calling error correction. If the sample barcode has a large number
    of reads, we'll send another batch script to handle these separately.
    Args:
        barcode: what RT barcode file for samples is being analysed
        directory: directory where sample barcode files are kept
        homopolymer_thresh: number of homopolymeric repeats, default =5 (now removed)
        big_mem: 'yes' or 'no'. if the txt file is a big one or not.
    """
    directory_path = pathlib.Path(directory)
    barcode_file = pathlib.Path(barcode)
    corrected_path = directory_path.joinpath("preprocessed_seq_corrected")
    pathlib.Path(corrected_path).mkdir(parents=True, exist_ok=True)
    big_output_directory_path = directory_path.joinpath("temp_big_ones")
    pathlib.Path(big_output_directory_path).mkdir(parents=True, exist_ok=True)
    # hopolA = "A" * homopolymer_thresh
    # hopolT = "T" * homopolymer_thresh
    # hopolC = "C" * homopolymer_thresh
    # hopolG = "G" * homopolymer_thresh
    raw_bc = pd.read_csv(
        barcode_file, delimiter="\t", skiprows=lambda x: (x != 0) and not x % 2
    )
    barcode_tab = pd.DataFrame()
    barcode_tab["full_read"] = raw_bc[
        (~raw_bc.iloc[:, 0].str.contains("N"))
        # & (~raw_bc.iloc[:, 0].str.contains(hopolA)) removed homopolymer requirement
        # & (~raw_bc.iloc[:, 0].str.contains(hopolG))
        # & (~raw_bc.iloc[:, 0].str.contains(hopolC))
        # & (~raw_bc.iloc[:, 0].str.contains(hopolT))
    ]
    barcode_tab["neuron_sequence"] = barcode_tab["full_read"].str[:32]
    barcode_tab["umi_sequence"] = barcode_tab["full_read"].str[32:46]
    # error correct neuron barcodes first (not as diverse as umi's, so whole table processed for big ones)
    int_file = corrected_path.joinpath(f"neuron_only_corrected_{barcode_file.stem}.pkl")
    if pathlib.Path(int_file).is_file() == False:
        neuron_bc_corrected = error_correct_sequence(
            reads=barcode_tab, sequence_type="neuron"
        )
        if big_mem == "yes":
            neuron_bc_corrected.to_pickle(
                int_file
            )  # if it's a big file, save intermediate in case job runs out of time
    # if big_mem == "no":
    #    corrected_sequences = error_correct_sequence(
    #        reads=neuron_bc_corrected, sequence_type="umi"
    #   )
    # elif big_mem == "yes":
    # corrected_sequences = neuron_bc_corrected
    if pathlib.Path(int_file).is_file():
        neuron_bc_corrected = pd.read_pickle(int_file)
    neuron_list = neuron_bc_corrected["corrected_sequences_neuron"].unique()
    neuron_bc_corrected["corrected_sequences_umi"] = "NA"
    n = 100  # max number of neuron barcodes to process at one time
    neuron_list_subsets = [
        neuron_list[i : i + n] for i in range(0, len(neuron_list), n)
    ]
    for sequence_str in neuron_list_subsets:
        # for sequence_str in neuron_list:
        neuron_bc_analysed = neuron_bc_corrected[
            neuron_bc_corrected["corrected_sequences_neuron"].isin(
                sequence_str
            )  # str.contains(sequence_str)
        ]
        chunk_analysed = error_correct_sequence(
            reads=neuron_bc_analysed, sequence_type="umi"
        )
        neuron_bc_corrected.update(chunk_analysed)
    corrected = sum(
        neuron_bc_corrected["umi_sequence"]
        != neuron_bc_corrected["corrected_sequences_umi"]
    )
    total = len(neuron_bc_corrected)
    print(f"Corrected {corrected} out of  {total} sequence counts for umis")
    if big_mem == "yes":
        os.remove(int_file)
    new_file = corrected_path.joinpath(f"corrected_{barcode_file.stem}.csv")
    neuron_bc_corrected.to_csv(new_file)
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
    if sequence_type == "neuron":
        print(
            f"Corrected {corrected} out of  {total} sequence counts for neuron barcodes"
        )
        tend = datetime.now()
        print(f"That took {tend - tstart}", flush=True)
    return barcode_tab


def join_tabs_and_split(directory, start_from_beginning):
    """
    Function to generate a table to check the amount of template switching across entire sample set, split into smaller chunks for processing in separate jobs
    Args:
        directory (str): path where the sequencing/pcr corrected sequences are kept
        start_from_beginning (str): 'yes' or 'no' whether want to start from combining files or not
    """
    n = 30000  # number of unique umi's to process at once per job
    reads_path = pathlib.Path(directory)
    template_dir = reads_path.joinpath("template_switching")
    pathlib.Path(template_dir).mkdir(parents=True, exist_ok=True)
    chunk_dir = template_dir.joinpath("chunks")
    pathlib.Path(chunk_dir).mkdir(parents=True, exist_ok=True)
    template_switch_script = "/camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Bash_scripts/template_chunks.sh"
    if start_from_beginning == "yes":
        barcode_sequences = pd.DataFrame(
            columns=["corrected_neuron", "corrected_UMI", "sample"]
        )
        print("starting combining samples into one big file", flush=True)
        for file in os.listdir(reads_path):
            barcode_file = reads_path / file
            if barcode_file.stem.startswith("corrected_"):
                bc_table = pd.read_csv(barcode_file)
                sample = barcode_file.stem.split("corrected_", 1)[1]
                new_tab = pd.DataFrame(
                    {
                        "corrected_neuron": bc_table["corrected_sequences_neuron"],
                        "corrected_UMI": bc_table["corrected_sequences_umi"],
                        "sample": sample,
                    }
                )
                barcode_sequences = pd.concat([barcode_sequences, new_tab])
        print(
            "finished combining samples into one big file, now sending jobs for umi cross barcode counting",
            flush=True,
        )
        barcode_sequences.to_csv(template_dir / "template_switching_all_seq.csv")
    if start_from_beginning == "no":
        print("reading full table combined", flush=True)
        barcode_sequences = pd.read_csv(template_dir / "template_switching_all_seq.csv")
    UMI_list = barcode_sequences["corrected_UMI"].unique()
    neuron_list_subsets = [UMI_list[i : i + n] for i in range(0, len(UMI_list), n)]
    iteration = 0
    for sequence_str in neuron_list_subsets:
        table_chunk = barcode_sequences[
            barcode_sequences["corrected_UMI"].isin(sequence_str)
        ]
        iteration = iteration + 1
        table_name = chunk_dir / f"chunk_{iteration}.csv"
        table_chunk.to_csv(table_name, index=False)
        command = f"sbatch {template_switch_script} {table_name}"
        print(command, flush=True)
        subprocess.Popen(shlex.split(command))


def switch_analysis(directory, chunk):
    """
    Function to perform analysis of which umi's are switching between different barcodes
    Args:
        directory(str): where the template path is
        chunk(str): table chunk of umi's to look at
    """
    outdir = pathlib.Path(directory)
    saving_path = outdir.joinpath("analysed_chunks")
    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
    barcode_sequences = pd.read_csv(chunk)
    chunk_path = pathlib.Path(chunk)
    which_chunk = chunk_path.stem.split("chunk_", 1)[1]
    tstart = datetime.now()
    UMI_list = barcode_sequences["corrected_UMI"].unique()
    template_switching_check = pd.DataFrame(
        {
            "UMI": UMI_list,
            "total": 0,
            "different_neurons": 0,
            "1st_abundant": 0,
            "2nd_abundant": 0,
            "sequence_of_1st": "A",
            "sample_of_1st": 0,
        }
    ).set_index("UMI")
    barcode_sequences["combined"] = (
        barcode_sequences["corrected_neuron"] + barcode_sequences["corrected_UMI"]
    )
    i = 1
    check_points = [10, 100, 1000, 5000, 10000, 20000, 50000]
    for umi in UMI_list:
        i = i + 1
        cross_barcode_counts = barcode_sequences[
            barcode_sequences.corrected_UMI.isin([umi])
        ].combined.value_counts()
        if len(cross_barcode_counts) > 1:
            second = cross_barcode_counts[1]
        else:
            second = 0
        template_switching_check.loc[umi, "total"] = cross_barcode_counts.sum()
        template_switching_check.loc[umi, "different_neurons"] = len(
            cross_barcode_counts
        )
        template_switching_check.loc[umi, "1st_abundant"] = cross_barcode_counts[0]
        template_switching_check.loc[umi, "2nd_abundant"] = second
        template_switching_check.loc[
            umi, "sequence_of_1st"
        ] = cross_barcode_counts.index[0]
        template_switching_check.loc[umi, "sample_of_1st"] = (
            barcode_sequences.loc[
                barcode_sequences["combined"] == cross_barcode_counts.index[0], "sample"
            ]
            .value_counts()
            .index[0]
        )
        if i in check_points:
            print(f"reached number {i} in {datetime.now()-tstart}", flush=True)
    print("finished umi cross barcode counting, saving file", flush=True)
    template_switching_check.to_csv(
        saving_path / f"template_switching_chunk_{which_chunk}.csv"
    )


def combine_switch_tables(template_sw_directory):
    """
    Function to combine tables from template switching analysis
    Args:
        template_sw_directory (str): path where individual analysed chunks are kept
    """
    dir_path = pathlib.Path(template_sw_directory)
    # saving_path = pathlib.Path(dir_path).parents[0]
    template_switching_check = pd.DataFrame(
        columns=[
            "UMI",
            "total",
            "different_neurons",
            "1st_abundant",
            "2nd_abundant",
            "sequence_of_1st",
            "sample_of_1st",
            "chunk",
        ]
    ).set_index("UMI")
    print("starting combining samples into one big file", flush=True)
    for file in os.listdir(dir_path):
        barcode_file = dir_path / file
        if barcode_file.stem.startswith("template_switching_chunk_"):
            bc_table = pd.read_csv(barcode_file)
            sample = barcode_file.stem.split("template_switching_chunk_", 1)[1]
            bc_table["chunk"] = sample
            template_switching_check = pd.concat([template_switching_check, bc_table])
    # template_switching_check.to_csv(
    #   saving_path / "combined_template_switching_chunks.csv"
    # )
    return template_switching_check


def combineUMIandBC(directory, UMI_cutoff=2, barcode_file_range=96):
    """
    Function to combine corrected barcodes and UMI's for each read and collect value counts.
    Also to split spike RNA from neuron barcodes, by whether contains N[24]ATCAGTCA (vs N[30][CT][CT] for neuron barcodes)
    Args:
        directory (str): path to temp file where the intermediate UMI and barcode clustered csv files are kept
        UMI_cutoff (int): threshold for minimum umi count per barcode. (default =2)
        barcode_file_range (int): the number of samples you want to loop through (default set for 96)
    """
    template_switch_abundance = 10  # min amount that if shared umi between neuron barcodes the most abundant neuron barcode umi must be more abundant by
    dir_path = pathlib.Path(directory)
    out_dir = dir_path.joinpath("Final_processed_sequences")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    # load template switching table and generate a list of UMI's to remove
    switching_tab = combine_switch_tables(
        template_sw_directory=dir_path / "template_switching/analysed_chunks"
    )
    # switching_tab =pd.read_csv(dir_path / "template_switching/combined_template_switching_chunks.csv")
    switches = switching_tab[switching_tab["different_neurons"] > 1]
    junk_umis = switches[
        switches["1st_abundant"] / switches["2nd_abundant"] <= template_switch_abundance
    ]["UMI"].tolist()
    switches = switches[
        switches["1st_abundant"] / switches["2nd_abundant"] > template_switch_abundance
    ].set_index("UMI")

    for i in range(barcode_file_range):
        barcode = f"BC{i+1}"
        sample_file = dir_path / f"corrected_{barcode}.csv"
        if os.path.isfile(sample_file):
            print("processing %s" % barcode, flush=True)
            sample_table = pd.read_csv(sample_file)
            # remove junk umi's
            sample_table = sample_table[
                ~sample_table["corrected_sequences_umi"].isin(junk_umis)
            ]

            sample_table["combined"] = (
                sample_table["corrected_sequences_neuron"]
                + sample_table["corrected_sequences_umi"]
            )
            pot_switches = sample_table[
                sample_table["corrected_sequences_umi"].isin(
                    list(switches.index.values)
                )
            ]
            # now remove umi sequences that are template switching events
            pot_switches["drop_or_not"] = pot_switches.apply(
                lambda x: "yes"
                if switches.loc[x["corrected_sequences_umi"]]["sequence_of_1st"]
                == x["combined"]
                and switches.loc[x["corrected_sequences_umi"]]["sample_of_1st"]
                != barcode
                else "yes"
                if switches.loc[x["corrected_sequences_umi"]]["sequence_of_1st"]
                != x["combined"]
                else "no",
                axis=1,
            )
            sample_table = sample_table.drop(
                pot_switches[pot_switches["drop_or_not"] == "yes"].index.tolist()
            )
            # separate and save spike in vs neuron barcodes
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
            print("finished %s" % barcode, flush=True)
            to_save_BC = out_dir / f"neuron_counts_{barcode}.csv"
            to_save_spike = out_dir / f"spike_counts_{barcode}.csv"
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
    all_seq = pd.DataFrame()
    for barcode_file in os.listdir(sorting_dir):
        if barcode_file.startswith("neuron_counts_"):
            print(f"reading barcode file {barcode_file}", flush=True)
            to_read = pd.read_csv(sorting_dir / barcode_file)
            sequences = to_read["sequence"]
            all_seq = pd.concat([all_seq, sequences])
    all_seq_unique = (
        all_seq.value_counts().rename_axis("sequence").reset_index(name="counts")
    )
    # tabulate counts of each barcode in each sample area

    samples = list(range(1, num_samples + 1))
    zeros = np.zeros(shape=(len(all_seq_unique["sequence"]), len(samples)))
    barcodes_across_sample = pd.DataFrame(zeros, columns=samples)
    barcodes_across_sample = barcodes_across_sample.set_index(
        all_seq_unique["sequence"]
    )

    for file in os.listdir(sorting_directory):
        barcode_file = sorting_dir / file
        if barcode_file.stem.startswith("neuron_counts_"):
            toread = pd.read_csv(barcode_file)
            sample = int(barcode_file.stem.split("neuron_counts_BC", 1)[1])
            for r, sequence in toread["sequence"].items():
                barcodes_across_sample.loc[sequence, sample] = toread["counts"][r]

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
    directory_path = pathlib.Path(directory)
    out_directory_path = pathlib.Path(outdir)
    spike_counts = pd.DataFrame(columns=["sample", "spike_count"])
    for sample in os.listdir(directory_path):
        if sample.startswith("spike_counts"):
            sample_name = sample.split("spike_counts_", 1)
            sample_name = sample_name[1][: -len(".csv")]
            sample_reading = pd.read_csv(sample)
            sample_reading["counts"] = sample_reading["counts"].astype("int")
            sum_counts = sample_reading["counts"].sum()
            new_row = pd.DataFrame(
                {"sample": sample_name, "spike_count": sum_counts}, index=[0]
            )
            spike_counts = pd.concat([spike_counts, new_row])
    lowest = min(spike_counts["spike_count"])
    spike_counts["normalisation_factor"] = spike_counts["spike_count"] / lowest
    print(spike_counts)
    for sample in os.listdir(directory_path):
        if sample.startswith("neuron_counts_"):
            sample_name = sample.split("neuron_counts_", 1)
            sample_name = sample_name[1][: -len(".csv")]
            sample_reading = pd.read_csv(sample)
            normalisation_factor = spike_counts.loc[
                spike_counts["sample"] == sample_name, "normalisation_factor"
            ].iloc[0]
            sample_reading["normalised_counts"] = (
                sample_reading["counts"] / normalisation_factor
            )
            name = out_directory_path / f"normalised_counts_{sample_name}.csv"
            sample_reading.to_csv(name)
