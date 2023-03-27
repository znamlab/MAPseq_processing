import os
import numpy as np
import pandas as pd


def barcode_matching(sorting_directory, num_samples):
    """
    Function to identify matching barcodes between samples
    Args: sorting directory =sorting directory where bowtieoutput files are saved
                num_samples = number of sample barcodes used
    """
    os.chdir(sorting_directory)
    all_seq = []

    for barcodefile in os.listdir(sorting_directory):
        if barcodefile.startswith("neuroncounts_"):
            print("reading barcode file %s" % barcodefile, flush=True)
            toread = pd.read_csv(barcodefile)
            sequences = toread["sequence"]
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
        for barcodefile in os.listdir(sorting_directory):
            if barcodefile.startswith("neuroncounts_"):
                toread = pd.read_csv(barcodefile)
                sample = int(barcodefile.split("neuroncounts_BC", 1)[1][: -len(".csv")])
                for r, sequence in toread["sequence"].items():
                    if sequence == barcode:
                        barcodes_across_sample.at[index, sample] = toread["counts"][r]
    print("finito")
    barcodes_across_sample.to_pickle("raw_barcodes_across_sample_higher_cutoff.pkl")
