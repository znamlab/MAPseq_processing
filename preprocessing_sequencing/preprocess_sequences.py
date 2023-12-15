# %%
import os
import numpy as np
import pandas as pd
from umi_tools import UMIClusterer
from datetime import datetime
import subprocess, shlex
import pathlib
import shutil
import subprocess
import gzip
from znamutils import slurm_it
import yaml


def load_parameters(directory="root"):
    """Load the parameters yaml file containing all the parameters required for
    preprocessing MAPseq data

    Args:
    directory (str): Directory where to load parameters from. Default 'root' for the
        default parameters (found in `mapseq_preprocessing/parameters.py`).

    Returns:
        dict: contents of parameters.yml
    """

    def flatten_dict(d):
        flattened_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened_dict[subkey] = subvalue
            else:
                flattened_dict[key] = value
        return flattened_dict

    if directory == "root":
        parameters_file = pathlib.Path(__file__).parent / "parameters.yml"
    else:
        parameters_file = pathlib.Path(directory) / "parameters.yml"
    with open(parameters_file, "r") as f:
        parameters = flatten_dict(yaml.safe_load(f))
    return parameters


def initialise_flz(
    MOUSE, PROJECT
):  # need to integrate --> currently not working and not sure why
    """Function to initalise flexilims so that you can update and also get paths
    Args:
        MOUSE: mouse id
        PROJECT: name of project in raw data folder

    """
    flm_ses = flz.get_flexilims_session(project_id=PROJECT)
    parent = flz.get_entity(
        datatype="sample", flexilims_session=flm_ses, name=f"{MOUSE}_Sequencing"
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=["FASTX-Toolkit"],
    slurm_options=dict(ntasks=1, time="72:00:00", mem="50G", partition="cpu"),
)
def split_samples(verbose=1, start_next_step=True):
    """Split raw fastq data according to sample barcodes

    This unzips raw fastq.gz files, cuts the two reads if needed (using `r1_part` and
    `r2_part`) and runs fastx_barcode_splitter.

    The output directory will contain a .txt file per barcode and a file named
    `barcode_splitter_log.txt` containing the summary output from the fastx function.

    Parameters read for param file:
        acq_id (str): Acquisition ID. Only file starting with this id will be unzipped
        barcode_file (str or Path): path to the file containing the list of barcodes
        raw_dir: (str or Path): Path to the folder containing the fastq.gz files. It
            should contain two files starting with acq_id, one for read 1 with `R1` in
            its name and one for read 2, with `R2` in its name
        output_dir (str or Path): [optional] Directory to save the output, if None,
            will save in the current working directory
        n_mismatch (int): [optional] number of mismatches accepted. Default to 1
        r1_part (None or (int, int)): [optional] part of the read 1 sequence to keep,
            None to keep the full read, [beginning, end] otherwise
        r2_part (int, int): [optional] same as r1_part but for read 2
       consensus_pos_start (int): start of expected consensus sequence for QC'ing reads as junk or not
       consensus_pos_end (int): end of expected consensus sequence for QC'ing reads as junk or not
       consensus_seq: consensus sequence that you would expect in reads

    Args:
        verbose (int): Level of feedback printed. 0 for nothing, 1 for steps,
            2 for steps and full output
        start_next_step (bool): If True (default), will start the next step in the
            pipeline (preprocess_reads)



    Returns:
        None
    """

    parameters = load_parameters(directory="root")
    parameters_file = pathlib.Path(__file__).parent / "parameters.yml"
    processed_dir = pathlib.Path(parameters["PROCESSED_DIR"])
    job_list = []
    # make a copy of the parameters file in the processed folder
    new_dir = (
            processed_dir
            / parameters["PROJECT"]
            / parameters["MOUSE"]
            / "Sequencing")
    new_dir.mkdir(parents=True, exist_ok=True)
    parameters_copy = new_dir / f"parameters.yml"
    shutil.copy(str(parameters_file), str(parameters_copy))
    #if more than one sequencing round, you will have more than one aqu_id in parameters, therefore you loop through and combine later
    for i in parameters['acq_id']:
        output_dir = (
            processed_dir
            / parameters["PROJECT"]
            / parameters["MOUSE"]
            / "Sequencing"
            / i
            / "BC_split"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        job = unzip_fastq(
        source_dir=parameters["RAW_DIR"],
        acq_id=i,
        overwrite=parameters["overwrite"],
        target_dir=str(output_dir), use_slurm=True,
        slurm_folder=parameters["SLURM_DIR"], scripts_name=f"unzip_and_split_{i}"
    )
        job_list.append(job)

    job_list = ":".join(map(str, job_list))
    if start_next_step:
        if verbose:
            print("Sending preprocess reads job", flush=True)
        preprocess_reads(
            directory=str(new_dir),
            barcode_range=parameters["barcode_range"],
            use_slurm=True,
            slurm_folder=parameters["SLURM_DIR"], job_dependency=job_list
        )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=["FASTX-Toolkit"],
    slurm_options=dict(ntasks=1, time="72:00:00", mem="50G", partition="cpu"),
)
def unzip_fastq(source_dir, acq_id, target_dir=None, overwrite=True, verbose=1):
    """Unzip fastq.gz files and sends for splitting into RT primer samples

    Args:
        source_dir (str or pathlib.Path): Path to the folder containing the
            fastq.gz files.
        acq_id (str): Acquisition ID. Only file starting with this id will be unzipped
        target_dir (str or pathlib.Path): Path to directory to write the output. If
            None (default), will write in source_dir
        overwrite (bool): Overwrite target if it already exists. Default to False
        verbose (int): 0 do not print anything. 1 print progress

    Returns:
        None
    """
    if verbose:
        tstart = datetime.now()
    source_dir = pathlib.Path(source_dir)
    assert source_dir.is_dir()

    if target_dir is None:
        target_dir = source_dir
    else:
        target_dir = pathlib.Path(target_dir)
        if not target_dir.is_dir():
            target_dir.mkdir(mode=774)
    parameters_loc = pathlib.Path(target_dir).parents[1]
    parameters = load_parameters(directory=str(parameters_loc))
    fastq_files = dict()
    for gz_file in source_dir.glob("{0}*.fastq.gz".format(acq_id)):
        target_file = target_dir / gz_file.stem
        if target_file.exists() and (not overwrite):
            raise IOError(f"{target_file} already exists. Use overwrite to replace")
        if verbose:
            t = datetime.now()
            print("Unzipping %s (%s)" % (gz_file, t.strftime("%H:%M:%S")))
        with gzip.open(gz_file, "rb") as source, open(target_file, "wb") as target:
            shutil.copyfileobj(source, target)
        fastq_files[target_file.stem] = target_file
    
        # make sure we have read1 and read2
    #if you've re-sequenced for more reads (and so have multiple fastq files), run barcode splitter separately, then 
    #combine
    read_files = dict()
    for read_number in [1, 2]:
        good_file = [r for r in fastq_files.keys() if "R%d" % read_number in r]
        if not good_file:
            raise IOError("Could not find read %d file" % read_number)
        elif len(good_file) > 1:
            raise IOError(
                "Found multiple files for read %d:\n%s" % (read_number, good_file)
            )
        else:
            read_files[read_number] = fastq_files[good_file[0]]
    r1_part = tuple(parameters["r1_part"])
    r2_part = tuple(parameters["r2_part"])
    consensus_pos = tuple(parameters["consensus_pos"])
    run_bc_splitter(
        read1_file=read_files[1],
        read2_file=read_files[2],
        barcode_file=parameters["BARCODE_DIR"],
        n_mismatch=parameters["n_mismatch"],
        r1_part=r1_part,
        r2_part=r2_part,
        output_dir=target_dir,
        verbose=1,
        consensus_pos=consensus_pos,
        consensus_seq=parameters["consensus_seq"],
    )
    # remove fastq files:
    for f in fastq_files.values():
        os.remove(f)

    if verbose:
        tend = datetime.now()
        print("That took %s" % (tend - tstart), flush=True)


def run_bc_splitter(
    read1_file,
    read2_file,
    barcode_file,
    n_mismatch=2,
    r1_part=None,
    r2_part=None,
    output_dir=None,
    verbose=1,
    consensus_pos=(37, 42),
    consensus_seq="GCGGC",
):
    """Split samples using Barcode splitter

    Format data and calls barcode splitter. It will also save the summary output of
    barcode splitter in 'barcode_splitter_log.txt' in the same directory as the other
    output file.

    Args:
        read1_file (str or Path): path to the fastq file of the first read
        read2_file (str or Path): path to the fastq file of the second read
        barcode_file (str or Path): path to the file containing the list of barcodes
        n_mismatch (int): [optional] number of mismatches accepted. Default to 1
        r1_part (None or (int, int)): [optional] part of the read 1 sequence to keep,
            None to keep the full read, [beginning, end] otherwise
        r2_part (int, int): [optional] same as r1_part but for read 2
        output_dir (str or Path): [optional] Directory to save the output, if None,
            will save in the current working directory
        verbose (int): Level of feedback printed. 0 for nothing, 1 for steps,
            2 for steps and full output
        consensus_pos (tuple)): start and end of expected consensus sequence for QC'ing
            reads as junk or not
        consensus_seq: consensus sequence that you would expect in reads

    Returns:
        None
    """
    if verbose:
        t = datetime.now()
        print(
            "Split sequence and merge reads (%s)" % t.strftime("%H:%M:%S"), flush=True
        )
    if output_dir is None:
        output_dir = os.getcwd()
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(mode=774)

    # barcode splitter expects a file with the sequence number on one line and the
    # sequence on the next. We will do that
    temp_file = output_dir / "barcode_splitter_input.fasta"
    with open(read1_file, "r") as read1, open(read2_file, "r") as read2, open(
        temp_file, "w"
    ) as target:
        # read on line out of 4
        n_reads = 1
        for il, (r1, r2) in enumerate(zip(read1, read2)):
            if il % 4 == 1:
                if r1[consensus_pos[0] : consensus_pos[1]] == consensus_seq:
                    # crop sequence if needed
                    if r1_part is not None:
                        r1 = r1[r1_part[0] : r1_part[1]]
                    if r2_part is not None:
                        r2 = r2[r2_part[0] : r2_part[1]]
                    # concatenate and write to temporary file
                    full_read = r1.strip() + r2.strip() + "\n"
                    target.write("> {0}\n{1}".format(n_reads, full_read))
                    n_reads += 1

    # split dataset according to inline indexes using fastx toolkit; this by default
    # allows up to 1 missmatch. we could go higher if we want, though maybe not
    # neccessary
    # now run barcode splitter on that
    if verbose:
        t = datetime.now()
        print("Barcode splitter (%s)" % t.strftime("%H:%M:%S"), flush=True)
    with open(temp_file, "r") as file_input:
        out = subprocess.run(
            [
                "fastx_barcode_splitter.pl",
                "--bcfile",
                str(barcode_file),
                "--prefix",
                str(output_dir) + os.path.sep,
                "--eol",
                "--suffix",
                ".txt",
                "--mismatches",
                str(n_mismatch),
            ],
            stdin=file_input,
            capture_output=True,
        )
    if out.stderr:
        raise IOError("Barcode splitter raised an error:\n{0}", out.stderr)

    log_file = output_dir / "barcode_splitter_log.txt"
    with open(log_file, "wb") as log:
        log.write(out.stdout)

    if verbose > 1:
        print(out.stdout.decode(), flush=True)
    if verbose:
        t = datetime.now()
        print(
            "Sample splitting done, removing temporary file (%s)"
            % t.strftime("%H:%M:%S")
        )
    os.remove(temp_file)


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="48:00:00", mem="350G", partition="hmem"),
)
def preprocess_reads(directory, barcode_range, max_file_size=100, extremely_large_job_size=6000):
    """Function to run UMI-tools to correct PCR and sequencing errors for UMI's and
    neuron barcodes separately, then take UMI counts for each barcode.

    Args:
        directory (str): directory where sample splitting occurred
        barcode_range (tuple): Id of the first and last barcodes to process (1, 91) e.g.
        max_file_size (float): maximum file size in MB at which the BC file is it to be
            processed using lower memory job. If higher than this, a separate job using
            a higher memory will be initiated (default 100)
        extremely_large_job_size (float): for files that are on the extremely large end, we will request max memory jobs (default 6GB)

    Returns:
        None.
    """
    # for loop to error correct neuron barcodes and umi's sequentially, but if file is
    # large, generate a separate script to process barcodes with higher memory, then
    # take each neuron barcode and correct umi's within each
    job_list = []

    directory_path = pathlib.Path(directory)
    parameters = load_parameters(directory=directory)
    BC_split_combined = pathlib.Path(directory_path/'BC_split_combined')
    BC_split_combined.mkdir(parents=True, exist_ok=True)
    all_files = os.listdir(directory_path/parameters['acq_id'][0]/'BC_split')
    print('combining barcode splitter output', flush=True)
    for file in all_files:
        with open(directory_path/parameters['acq_id'][0]/'BC_split'/file, "r") as file_1:
            file_1_contents = file_1.read() #combine BC split output for different read folders
        for n, i in enumerate(parameters['acq_id']):
                if n > 0:
                    with open(directory_path/i/'BC_split'/file, "r") as file_next:
                        file_next_contents = file_next.read()
                    file_1_contents = file_1_contents + file_next_contents
        out_path =  BC_split_combined/file
        with open(out_path, "w") as output_file:
            output_file.write(file_1_contents)
        del file_1, file_next, file_1_contents, file_next_contents, output_file
    barcode_range = tuple(parameters["barcode_range"])
    day = datetime.now().strftime("%Y-%m-%d")
    slurm_folder = pathlib.Path(parameters["SLURM_DIR"] + f"/{day}")
    slurm_folder.mkdir(parents=True, exist_ok=True)
    print('finished combining barcode splitter output, now sending jobs for neuron sequence correcton', flush=True)
    # Correct neuron barcodes first
    for x in range(barcode_range[0], barcode_range[1] + 1, 1):
        barcode_num = "BC" + str(x) + ".txt"
        barcode_file = BC_split_combined.joinpath(barcode_num)
        print(f'looking at {barcode_num}', flush=True)
        if not pathlib.Path(barcode_file).is_file():
            print(f"BC file {barcode_file} not found")
            continue

        fsize = barcode_file.stat().st_size / 1024.0 / 1024.0  # size in MB
        if not fsize:
            print(f"Nothing in {barcode_file}")
            continue
        
        kwargs = dict(barcode_file=str(barcode_file), directory=str(BC_split_combined))
        # for big files, start a separate job
        if fsize > max_file_size and fsize <= extremely_large_job_size:
            kwargs.update(
                use_slurm=True,
                slurm_folder=slurm_folder,
                scripts_name=f"neuron_correction_BC{x}",
            )
        if fsize > extremely_large_job_size:
            kwargs.update(
                use_slurm=True,
                slurm_folder=slurm_folder,
                scripts_name=f"neuron_correction_BC{x}", slurm_options=dict(mem="250G")
            )
        job = process_neuron_barcodes(**kwargs)
        if fsize > max_file_size:
            job_list.append(job)
    print(f"Started {len(job_list)} jobs for big files", flush=True)

    # then correct umi's
    job_list = ":".join(map(str, job_list))
    correct_all_umis(
        directory=str(directory_path),
        use_slurm=True,
        slurm_folder=parameters["SLURM_DIR"],
        job_dependency=job_list,
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="72:00:00", mem="64G", partition="cpu"),
)
def process_neuron_barcodes(barcode_file, directory, redo=False):
    """Function to process the barcode sequences, removing homopolymers before
    calling error correction.

    Args:
        barcode (str): what RT barcode file for samples is being analysed
        directory: directory where sample barcode files are kept
        redo (bool): whether to redo the analysis if it has already been done. Default
            False
    """
    directory = pathlib.Path(directory)
    barcode_file = pathlib.Path(barcode_file)
    temp_output = directory.parent / "temp_big_ones"
    temp_output.mkdir(parents=True, exist_ok=True)
    barcode = barcode_file.stem

    raw_bc = pd.read_csv(
        barcode_file, delimiter="\t", skiprows=lambda x: (x != 0) and not x % 2
    )
    barcode_tab = pd.DataFrame()
    barcode_tab["full_read"] = raw_bc[(~raw_bc.iloc[:, 0].str.contains("N"))].copy()
    del raw_bc  # not needed anymore
    barcode_tab["neuron_sequence"] = barcode_tab["full_read"].str[:32]
    barcode_tab["umi_sequence"] = barcode_tab["full_read"].str[32:46]
    # error correct neuron barcodes first (not as diverse as umi's, so whole table
    # processed for big ones)
    int_file = temp_output.joinpath(f"neuron_only_corrected_{barcode}.pkl")
    if pathlib.Path(int_file).is_file() and not redo:
        print("Already have intermediate pkl file", flush=True)
        #neuron_bc_corrected = pd.read_pickle(int_file)
    else:
        print(f"Looking at {barcode}")
        neuron_bc_corrected = error_correct_sequence(
            reads=barcode_tab, sequence_type="neuron"
        )
        neuron_bc_corrected.to_pickle(int_file)
        print("Saved intermediate pkl file", flush=True)

@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="48:00:00", mem="350G", partition="hmem"),
)
def correct_all_umis(directory, barcode_per_batch=50000):
    """Function to correct umi sequences using UMI tools directional adjacency approach

    This will read all the barcode files, already corrected for neuron barcodes, and
    correct the umi sequences using UMI tools directional adjacency approach.

    To avoid crashing nemo, UMI are processed by batch. The output is saved in a
    temporary pkl file. Then a separate job will collate all the results.

    Args:
        directory (str): directory where sample barcode files are kept
        barcode_per_batch (int): number of barcodes to process at once per job

    Returns:
        None
    """
    directory_path = pathlib.Path(directory)

    parameters = load_parameters(directory)
    day = datetime.now().strftime("%Y-%m-%d")
    slurm_folder = pathlib.Path(parameters["SLURM_DIR"] + f"/{day}")
    slurm_folder.mkdir(parents=True, exist_ok=True)
    temp_output = directory_path / "temp_big_ones"
    assert temp_output.is_dir(), "Corrected path not found"
    job_ids = []

    # Iterates on all neuron_only files:
    for neuron_file in temp_output.glob("neuron_only_corrected_*.pkl"):
        neuron_bc_corrected = pd.read_pickle(neuron_file)
        barcode = neuron_file.stem.split("neuron_only_corrected_", 1)[1]
        # Correct umi's
        # We process by bunch of neuron barcodes at a time, as umi's are more diverse
        # We will do batch of at least 50000 barcodes at a time, but won't split data
        # coming from the same neuron obviously
        neuron_list = neuron_bc_corrected["corrected_sequences_neuron"].value_counts()
        to_process = list(neuron_list.index)
        batch = []
        n_in_batch = 0
        batch_id = 0
        while to_process:
            batch.append(to_process.pop())
            n_in_batch += neuron_list[batch[-1]]
            if n_in_batch >= barcode_per_batch:
                output_file = temp_output / f"{barcode}_batch_{batch_id}.pkl"
                # start the job
                job_id = correct_umi_sequences(
                    barcode_file=str(neuron_file),
                    neurons_to_process=batch,
                    output_file=str(output_file),
                    use_slurm=True,
                    scripts_name=f"UMI_{barcode}_correction_{batch_id}",
                    slurm_folder=str(slurm_folder),
                )
                job_ids.append(job_id)
                batch = []
                n_in_batch = 0
                batch_id += 1

    # Finally start the next job which will collate all the results once the jobs have
    # finished
    job_ids = ",".join(map(str, job_ids))
    collate_error_correction_results(
        directory=directory,
        temp_output=str(temp_output),
        use_slurm=True,
        scripts_name=f"UMI_correction_collate",
        slurm_folder=parameters["SLURM_DIR"],
        job_dependency=job_ids,
    )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="72:00:00", mem="64G", partition="cpu"),
)
def correct_umi_sequences(barcode_file, neurons_to_process, output_file):
    """Function to correct umi sequences using UMI tools directional adjacency approach

    This is a very simple wrapper around the error_correct_sequence function, which
    is called with `sequence_type="umi"`.
    It is here to start independent jobs for each batch of umi's to process.

    Args:
        barcode_file (str): path to the file containing the list of barcodes
        neurons_to_process (list): list of neuron barcodes to process
        output_file (str): path to output file

    Returns:
        batch_corrected (pandas.DataFrame): table containing corrected umi sequences
    """
    neuron_bc_corrected = pd.read_pickle(barcode_file)
    reads = neuron_bc_corrected[
        neuron_bc_corrected["corrected_sequences_neuron"].isin(neurons_to_process)
    ].copy()

    del neuron_bc_corrected
    print("Correcting umi sequences for batch", flush=True)
    batch_corrected = error_correct_sequence(reads=reads, sequence_type="umi")
    print("Finished correcting umi sequences for batch", flush=True)
    batch_corrected.to_pickle(output_file)
    print(f"Saved batch corrected umi table for {barcode_file}")
    return batch_corrected


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="72:00:00", mem="350G", partition="hmem"),
)
def collate_error_correction_results(directory, temp_output):
    """Collate the results of the error correction jobs

    This will read all the files in the directory and concatenate them into one table.
    It will then delete all the individual files.

    Args:
        directory (str): directory where sample barcode files are kept
        temp_output (str): directory where the temporary files are kept


    Returns:
        None
    """
    directory = pathlib.Path(directory)
    parameters = load_parameters(directory=directory)
    temp_output = pathlib.Path(temp_output)
    # barcode_file = pathlib.Path(barcode_file)
    # barcode = barcode_file.stem

    corrected_path = directory / "preprocessed_seq_corrected"
    corrected_path.mkdir(parents=True, exist_ok=True)

    for neuron_file in temp_output.glob("neuron_only_corrected_*.pkl"):
        barcode = neuron_file.stem.split("neuron_only_corrected_", 1)[1]
        print(f'looking at {barcode}', flush=True)
        output_file = corrected_path.joinpath(f"corrected_{barcode}.csv")
        if output_file.is_file():
            print(f"Warning: Output file {output_file} already exists. Overwriting", flush= True)

        # read all the files in the directory and concatenate them into one table
        all_files = temp_output.glob(f"{barcode}_batch_*.pkl")
        all_tables = []
        for f in all_files:
            all_tables.append(pd.read_pickle(f))
        if not all_tables:
            print(f'There are no {barcode} batch files', flush=True)
            continue
        all_tables = pd.concat(all_tables)
        all_tables.to_csv(output_file)
        print("Saved corrected umi table", flush=True)

        # Delete the barcode only intermediate files
        int_file = temp_output.joinpath(f"neuron_only_corrected_{barcode}.pkl")
        if int_file.is_file():
        #    os.remove(int_file)
            print("Removed intermediate pkl file", flush=True)
        # delete the UMI batch files
        #for f in all_files:
        #    os.remove(f)
        print("Removed all batch files", flush=True)

    # start the next job
    join_tabs_and_split(
        directory=str(directory),
        use_slurm=True,
        slurm_folder=parameters["SLURM_DIR"],
    )


def error_correct_sequence(reads, sequence_type):
    """Function to perform error correction using UMI tools directional adjacency
    approach

    Args:
        reads (pandas.DataFrame): table that has the reads that your interested in (has
            to be with specific column headers seen in preprocess_reads function, i.e
            `full_read`, `neuron_sequence`, `umi_sequence`)
        sequence_type (str): 'neuron' or 'umi'. Are the sequences neuron barcodes or
            umi's?
    Returns: table containing corrected sequences
    """
    tstart = datetime.now()
    barcode_tab = reads  # why do you call it reads then?
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


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="48:00:00", mem="350G", partition="hmem"),
)
def join_tabs_and_split(directory):
    """
    Function to generate a table to check the amount of template switching across entire
    sample set, split into smaller chunks for processing in separate jobs

    Args:
        directory (str): path where the sequencing/pcr corrected sequences are kept
        num_umi (int): number of unique umi's to process at once per job
    """
    reads_path = pathlib.Path(directory) / "preprocessed_seq_corrected"
    template_dir = pathlib.Path(directory) / "template_switching"
    pathlib.Path(template_dir).mkdir(parents=True, exist_ok=True)
    chunk_dir = template_dir.joinpath("chunks")
    parameters = load_parameters(directory=directory)
    day = datetime.now().strftime("%Y-%m-%d")
    slurm_folder = parameters["SLURM_DIR"] + f"/{day}"
    pathlib.Path(slurm_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(chunk_dir).mkdir(parents=True, exist_ok=True)

    if pathlib.Path(template_dir / "template_switching_all_seq.csv").is_file():
        print("Reading full table combined", flush=True)
        barcode_sequences = pd.read_csv(template_dir / "template_switching_all_seq.csv")
    else:
        barcode_sequences = pd.DataFrame(
            columns=["corrected_neuron", "corrected_UMI", "sample"]
        )
        print("Starting combining samples into one big file", flush=True)
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
            "Finished combining samples into one big file, now sending jobs for umi cross barcode counting",
            flush=True,
        )
        barcode_sequences.to_csv(template_dir / "template_switching_all_seq.csv")

    UMI_list = barcode_sequences["corrected_UMI"].unique()
    neuron_list_subsets = [
        UMI_list[i : i + parameters["num_umi"]]
        for i in range(0, len(UMI_list), parameters["num_umi"])
    ]
    job_list = []
    for iteration, sequence_str in enumerate(neuron_list_subsets):
        table_chunk = barcode_sequences[
            barcode_sequences["corrected_UMI"].isin(sequence_str)
        ]
        table_name = chunk_dir / f"chunk_{iteration}.csv"
        table_chunk.to_csv(table_name, index=False)
        table_chunk_job = switch_analysis(
            directory=str(template_dir),
            chunk=str(table_name),
            use_slurm=True,
            slurm_folder=slurm_folder,
            scripts_name=f"template_switching_analysis_{iteration}",
        )
        job_list.append(table_chunk_job)
    job_list = ":".join(map(str, job_list))
    print(
        "Finished template switching analysis. After jobs finished, open 'determine_UMI_cutoff_and_template_switching_thresholds.ipynb' notebook to check UMI and template sharing distribution and alter parameters UMI cutoff and if necessary template_switch_abundance threshold in yaml file in processed directory, then running final preprocessing function"
    )
    # combineUMIandBC(
    #   directory=str(template_dir.parent),
    #  use_slurm=True,
    # slurm_folder=parameters["SLURM_DIR"],
    # job_dependency=job_list,
    # )


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="72:00:00", mem="30G", partition="cpu"),
)
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
    print("Starting combining samples into one big file", flush=True)
    for file in os.listdir(dir_path):
        barcode_file = dir_path / file
        if barcode_file.stem.startswith("template_switching_chunk_"):
            bc_table = pd.read_csv(barcode_file)
            sample = barcode_file.stem.split("template_switching_chunk_", 1)[1]
            bc_table["chunk"] = sample
            template_switching_check = pd.concat([template_switching_check, bc_table])
    return template_switching_check


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="72:00:00", mem="250G", partition="cpu"),
)
def combine_UMI_and_BC(directory):
    """
    N.B. visualise UMI count distribution first before running this step, as it's necessary to determin UMI cut-off parameters
    Function to combine corrected barcodes and UMI's for each read and collect value counts.
    Also to split spike RNA from neuron barcodes, by whether contains N[24]ATCAGTCA (vs N[30][CT][CT] for neuron barcodes)
    Args:
        directory (str): path to temp file where the intermediate UMI and barcode clustered csv files are kept
        UMI_cutoff (int): threshold for minimum umi count per barcode. (default =2)
        barcode_file_range (int): the number of samples you want to loop through (default set for 96)
        template_switch_abundance (int): min amount that if shared umi between neuron barcodes the most abundant neuron barcode umi must be more abundant by (default 10)
        neuron_bc_length (int): length of neuron barcode (this is 32nt for original MAPseq sindbis virus library)
        rom start of read 1 neuron barcode sequence, the sequence that differentiates a neuron barcode as belonging to spike-in RNA
        spike_in_identifier: "^.{24}ATCAGTCA" from start of read 1 neuron barcode sequence, the sequence that differentiates a neuron barcode as belonging to spike-in RNA
        neuron_identifier: "^.{30}[CT][CT]" from start of read 1 neuron barcode sequence, the sequence that differentiates a neuron barcode as belonging to actual neurons


    """
    parameters = load_parameters(directory=directory)
    spike_in_identifier = parameters["spike_in_identifier"]
    neuron_identifier = parameters["neuron_identifier"]
    UMI_cutoff = parameters["UMI_cutoff"]
    barcode_file_range = tuple(parameters["barcode_range"])
    template_switch_abundance = parameters["template_switch_abundance"]
    neuron_bc_length = parameters["neuron_bc_length"]
    dir_path = pathlib.Path(directory)
    out_dir = dir_path.joinpath("final_processed_sequences")
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

    for i in range(barcode_file_range[0], barcode_file_range[1] + 1, 1):
        barcode = f"BC{i+1}"
        sample_file = dir_path / f"preprocessed_seq_corrected/corrected_{barcode}.csv"
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
            if pot_switches.empty == False:
                # now remove umi sequences that are template switching events
                print(f"correcting template switching events for {barcode}", flush=True)
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
            elif pot_switches.empty == False:
                print(f"no template switches found for {barcode}", flush=True)
            # separate and save spike in vs neuron barcodes
            print(f"collapsing umi counts for {barcode}", flush=True)
            spike_in = sample_table[
                sample_table["combined"].str.contains(spike_in_identifier) == True
            ].rename_axis("sequence")
            neurons = sample_table[
                sample_table["combined"].str.contains(neuron_identifier) == True
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
            spike_counts["barcode"] = spike_counts["sequence"].str[:neuron_bc_length]
            neuron_counts["barcode"] = neuron_counts["sequence"].str[:neuron_bc_length]
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
            print(f"finished {barcode}", flush=True)
            to_save_BC = out_dir / f"neuron_counts_{barcode}.csv"
            to_save_spike = out_dir / f"spike_counts_{barcode}.csv"
            spike_counts.to_csv(to_save_spike)
            neuron_counts.to_csv(to_save_BC)
        else:
            print(f"file not there for {barcode}")
    print(f"Finished final pre-preprocessing step at {datetime.now()}")


@slurm_it(
    conda_env="MAPseq_processing",
    module_list=None,
    slurm_options=dict(ntasks=1, time="48:00:00", mem="50G", partition="cpu"),
)
def barcode_matching(sorting_directory):
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
    parameters = load_parameters(directory=sorting_dir.parent)
    barcode_range = tuple(parameters["barcode_range"])
    samples = list(range(barcode_range[0], barcode_range[1] + 1, 1))
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
