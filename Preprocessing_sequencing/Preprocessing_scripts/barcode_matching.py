import os
import numpy as np
import pandas as pd

def barcode_matching(sorting_directory, num_samples):
    """
    Function to identify matching barcodes between samples
    Args: sorting directory =sorting directory where bowtieoutput files are saved
                num_samples = number of sample barcodes used
    """
    #sorting_directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/140422_full_run/230422/sorting'
    os.chdir(sorting_directory)
    all_seq = []

    for barcodefile in os.listdir(sorting_directory):
            if barcodefile.startswith("normalised_counts_BC"):
                print('reading barcode file %s' %barcodefile, flush=True)
                toread = pd.read_csv(barcodefile)
                sequences = toread['sequence']
                all_seq.append(sequences)

    all_seq = pd.DataFrame(all_seq)
    bla = all_seq.to_numpy().flatten()
    all_seq = [x for x in bla if str(x) != 'nan']
    all_seq_unique = np.unique(all_seq)

    #tabulate counts of each barcode in each sample area

    samples = list(range(1, num_samples+1))
    barcodes_across_sample = pd.DataFrame(columns = samples, dtype = int)

    index = -1
    for barcode in all_seq_unique:
        index += 1
        for barcodefile in os.listdir(sorting_directory):
            if barcodefile.startswith("normalised_counts_BC"):
                toread = pd.read_csv(barcodefile)
                sample = int(barcodefile.split('normalised_counts_BC', 1)[1][:-len('.csv')])
                for r, sequence in toread['sequence'].items():
                    if sequence == barcode:
                        barcodes_across_sample.at[index, sample]= toread['normalised_counts'][r]
    print('finito')
    barcodes_across_sample.to_pickle("barcodes_across_sample.pkl")
