import os
import pandas as pd
from umi_tools import UMIClusterer
from datetime import datetime
import subprocess, shlex
from pathlib import Path


def batch_collapse_barcodes(newdir, barcodenum, numberdivisions):
    """
    Function to send a load of slurm jobs for bigger files to process in chunks of 5mil reads, so it won't take a lifetime to process.

    Args:
        newdir: where to big intermediate files are kept
        barcodenum: RT sample barcode identifier
        numberdivisions: the number of chunks you have for each bigger file
    """
    os.chdir(newdir)
    script_path = "/camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Bash_scripts/process_big_onesBarcode.sh"
    for x in range(numberdivisions):
        toread = "NeuronBCintermediate_%s_%s.csv" % (barcodenum, x + 1)
        command = f"sbatch {script_path} {toread}"
        print(command)
        subprocess.Popen(shlex.split(command))


def error_correct_barcodes(
    barcodefile, directory, bigonesorting_dir, homopolymerthresh=5
):
    """
    Function to run UMI-tools to correct PCR and sequencing errors, then take UMI counts for each barcode.

    Args:
        barcodefile: sample to read
        directory: directory where sample splitting occurred
        bigonesorting_dir: where the large intermediate files are stored
        homopolymerthres: we use a cut off for homopolymeric repeats (a common sequencing error). default=5
    """
    os.chdir(directory)
    # data_path = Path(directory) / barcodefile.strip()
    hopolA = "A" * homopolymerthresh
    hopolT = "T" * homopolymerthresh
    hopolC = "C" * homopolymerthresh
    hopolG = "G" * homopolymerthresh
    suffix = ".txt"
    barcodefile = barcodefile.strip()
    barcodenum = barcodefile[: -len(suffix)]
    # barcodenum = data_path.stem
    print("Starting barcode collapsing for %s" % barcodenum)
    tstart = datetime.now()
    raw_bc = pd.read_csv(
        barcodefile, delimiter="\t", skiprows=lambda x: (x != 0) and not x % 2
    )
    barcode_tab = pd.DataFrame()
    barcode_tab["full_read"] = raw_bc[
        (~raw_bc.iloc[:, 0].str.contains("N"))
        & (~raw_bc.iloc[:, 0].str.contains(hopolA))
        & (~raw_bc.iloc[:, 0].str.contains(hopolG))
        & (~raw_bc.iloc[:, 0].str.contains(hopolC))
        & (~raw_bc.iloc[:, 0].str.contains(hopolT))
    ]
    barcode_tab["neuron_bc"] = barcode_tab["full_read"].str[:32]

    size = os.stat(barcodefile).st_size
    if (
        size > 1000000000
    ):  # if the size of the barcode table is huge, we'll split it into smaller chunks for processing, then cluster together again at end
        numberdivisions = round(barcode_tab.shape[0] / 2000000)
        sequences = barcode_tab
        num = 0
        for i in range(numberdivisions):
            if num != numberdivisions - 1:
                df_short = sequences.iloc[:2000000, :]
                sequences = sequences.iloc[2000000:, :]
            else:
                df_short = sequences
            newtmpfile = "temp/bigones/NeuronBCintermediate_%s_%s.csv" % (
                barcodenum,
                num + 1,
            )
            df_short.to_csv(newtmpfile)
            num = num + 1
        newdir = bigonesorting_dir
        batch_collapse_barcodes(
            newdir=newdir, barcodenum=barcodenum, numberdivisions=numberdivisions
        )
        print("split %s into %s repetitions" % (barcodenum, numberdivisions))
    else:
        clusterer = UMIClusterer(cluster_method="directional")
        barcode_tab["bytneuronBC"] = [x.encode() for x in barcode_tab.neuron_bc]
        UMIcounts = barcode_tab["bytneuronBC"].value_counts()
        UMIdict = UMIcounts.to_dict()
        clustered_umis = clusterer(UMIdict, threshold=1)
        mydict = {}
        for x in clustered_umis:
            correct = x[0]  # correct (more frequent) UMI
            if len(x) > 1:
                for each in x[1:]:
                    mydict[each] = correct
            mydict[correct] = correct
        barcode_tab["corrected_neuronBC"] = [mydict[x] for x in barcode_tab.bytneuronBC]
        barcode_tab["corrected_neuronBC"] = [
            x.decode() for x in barcode_tab.corrected_neuronBC
        ]
        corrected = sum(barcode_tab.neuron_bc != barcode_tab.corrected_neuronBC)
        total = len(barcode_tab)
        correct_neuronbc = barcode_tab["corrected_neuronBC"]
        newfile = "temp/neuronBCcorrected_%s.csv" % barcodenum
        correct_neuronbc.to_csv(newfile)
        print(
            "Corrected %s out of  %s neuron barcode counts for %s"
            % (corrected, total, barcodefile)
        )
    tend = datetime.now()
    print("That took %s" % (tend - tstart), flush=True)


def error_correcting(directory, barcodefile, threshold):
    """
    Function to correct PCR and sequencing errors in reads and deduplicate umi's

    Args:
        directory: where the file is
        barcodefile: pandas dataframe containing reads
        threshold: edit distance for umi/barcode
    """
    clusterer = UMIClusterer(cluster_method="directional")
    barcode_tab["bytneuronBC"] = [x.encode() for x in barcode_tab.neuron_bc]
    UMIcounts = barcode_tab["bytneuronBC"].value_counts()
    UMIdict = UMIcounts.to_dict()
    clustered_umis = clusterer(UMIdict, threshold=1)
    mydict = {}
        for x in clustered_umis:
            for each in x:
                mydict[each] = x[0]

        barcode_tab["corrected_neuronBC"] = [
            mydict[x].decode() for x in barcode_tab.bytneuronBC
        ]
    corrected = sum(barcode_tab.neuron_bc != barcode_tab.corrected_neuronBC)
    total = len(barcode_tab)
    correct_neuronbc = barcode_tab["corrected_neuronBC"]
    newfile = "temp/neuronBCcorrected_%s.csv" % barcodenum
    correct_neuronbc.to_csv(newfile)
    print(
        "Corrected %s out of  %s neuron barcode counts for %s"
        % (corrected, total, barcodefile)
    )
    tend = datetime.now()
    print("That took %s" % (tend - tstart), flush=True)


def error_correct_chunk(directory, toread):
    """
    Function to process the chunks of 5mil reads from bigger files, to correct sequencing errors, so it won't take a lifetime to process.

    Args:
        toread: split barcode csv file to read
        directory: where files are
    """
    os.chdir(directory)
    barcode_tab = pd.read_csv(toread)
    suffix = ".csv"
    barcodenum = toread[: -len(suffix)]
    print("Starting barcode collapsing for chunk %s" % barcodenum, flush=True)
    tstart = datetime.now()
    barcode_tab["bytneuronBC"] = [x.encode() for x in barcode_tab.neuron_bc]
    UMIcounts = barcode_tab["bytneuronBC"].value_counts()
    UMIdict = UMIcounts.to_dict()
    clusterer = UMIClusterer(cluster_method="directional")
    clustered_umis = clusterer(UMIdict, threshold=1)
    mydict = {}
    for x in clustered_umis:
        for each in x:
            mydict[each] = x[0]

    barcode_tab["corrected_neuronBC"] = [
        mydict[x].decode() for x in barcode_tab.bytneuronBC
    ]

    corrected = sum(barcode_tab.neuron_bc != barcode_tab.corrected_neuronBC)
    total = len(barcode_tab)
    correct_neuronbc = barcode_tab["corrected_neuronBC"]
    newfile = f"neuronBCcorrected_{barcodenum}.csv"
    correct_neuronbc.to_csv(newfile)
    tend = datetime.now()
    print(
        f"Corrected {corrected} out of {total} neuron barcode counts for {barcodenum}"
    )
    print(f"That took {tend - tstart}", flush=True)


def combine_the_chunks(directory, outdirectory, barcodefile):
    """
    Function to combine the smaller files for big files that were processed separately

    Args:
        directory (.csv): location of where the files are
        barcodefile (BC... string): which one you want to look at (can this make to read from list, but at the moment, it's taking a while to process chunks in prev steps, so can't do them all)
    """
    os.chdir(directory)
    umicombined = pd.DataFrame(columns=["Unnamed: 0", "corrected_umi"])
    bccombined = pd.DataFrame(columns=["Unnamed: 0", "corrected_neuronBC"])
    for file in os.listdir(directory):
        if file.startswith("neuronBCcorrected_NeuronBCintermediate_%s" % barcodefile):
            barcodenumbers = file.split("neuronBCcorrected_NeuronBCintermediate_", 1)
            barcodenumbers = barcodenumbers[1][: -len(".csv")]
            umifiletoread = "UMIs_corrected_UMIintermediate_%s.csv" % barcodenumbers
            umicombined = pd.concat([umicombined, pd.read_csv(umifiletoread)])
            bccombined = pd.concat([bccombined, pd.read_csv(file)])
    combined = pd.concat([bccombined, umicombined], axis=1)
    combined["combined"] = combined["corrected_neuronBC"] + combined["corrected_umi"]
    spikein = combined[
        combined["combined"].str.contains("^.{24}ATCAGTCA") == True
    ].rename_axis("sequence")
    neurons = combined[
        combined["combined"].str.contains("^.{30}[CT][CT]") == True
    ].rename_axis("sequence")
    neuroncounts = (
        neurons["combined"]
        .value_counts()
        .rename_axis("sequence")
        .reset_index(name="counts")
    )
    counts_spike = (
        spikein["combined"]
        .value_counts()
        .rename_axis("sequence")
        .reset_index(name="counts")
    )
    # only take umi counts greater or equal to 2
    neuroncounts = neuroncounts[neuroncounts["counts"] >= 2]
    counts_spike = counts_spike[counts_spike["counts"] >= 2]
    counts_spike["barcode"] = counts_spike["sequence"].str[:32]
    neuroncounts["barcode"] = neuroncounts["sequence"].str[:32]
    spikeneuron = (
        counts_spike["barcode"]
        .value_counts()
        .rename_axis("sequence")
        .reset_index(name="counts")
    )
    neuroncounts = (
        neuroncounts["barcode"]
        .value_counts()
        .rename_axis("sequence")
        .reset_index(name="counts")
    )
    os.chdir(outdirectory)
    print("finished processing %s" % barcodefile, flush=True)
    tosaveBC = "neuroncounts_%s.csv" % barcodefile
    tosavespike = "spikecounts_%s.csv" % barcodefile
    spikeneuron.to_csv(tosavespike)
    neuroncounts.to_csv(tosaveBC)


def combineUMIandBC(directory, outdirectory, barcodefilerange=96):
    """
    Function to combine corrected barcodes and UMI's for each read and collect value counts.
    Also to detect degree of template switching between reads by seeing if UMI is shared by more than one barcode
    Also to split spike RNA from neuron barcodes, by whether contains N[24]ATCAGTCA (vs N[32]CTCT for neuron barcodes)
    Args:
        directory: temp file where the intermediate UMI and barcode clustered csv files are kept
        barcodefilerange: the number of samples you want to loop through (default set for 96)
    """
    os.chdir(directory)
    for i in range(barcodefilerange):
        os.chdir(directory)
        num = i + 1
        barcode = "BC%s" % num
        neuronfile = "neuronBCcorrected_%s.csv" % barcode
        umifile = "UMIs_corrected_%s.csv" % barcode
        if os.path.isfile(neuronfile) and os.path.isfile(umifile):
            print("processing %s" % barcode, flush=True)
            combined = pd.concat(
                [pd.read_csv(neuronfile), pd.read_csv(umifile)], axis=1
            )
            combined["combined"] = (
                combined["corrected_neuronBC"] + combined["corrected_umi"]
            )
            spikein = combined[
                combined["combined"].str.contains("^.{24}ATCAGTCA") == True
            ].rename_axis("sequence")
            neurons = combined[
                combined["combined"].str.contains("^.{30}[CT][CT]") == True
            ].rename_axis("sequence")
            neuroncounts = (
                neurons["combined"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            counts_spike = (
                spikein["combined"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            # only take umi counts greater or equal to 2
            neuroncounts = neuroncounts[neuroncounts["counts"] >= 2]
            counts_spike = counts_spike[counts_spike["counts"] >= 2]
            counts_spike["barcode"] = counts_spike["sequence"].str[:32]
            neuroncounts["barcode"] = neuroncounts["sequence"].str[:32]
            spikeneuron = (
                counts_spike["barcode"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            neuroncounts = (
                neuroncounts["barcode"]
                .value_counts()
                .rename_axis("sequence")
                .reset_index(name="counts")
            )
            os.chdir(outdirectory)
            print("finished %s" % barcode, flush=True)
            tosaveBC = "neuroncounts_%s.csv" % barcode
            tosavespike = "spikecounts_%s.csv" % barcode
            spikeneuron.to_csv(tosavespike)
            neuroncounts.to_csv(tosaveBC)
        else:
            print("both not there for %s" % barcode)


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
