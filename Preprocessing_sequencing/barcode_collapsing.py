import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import (draw, DiGraph, Graph)
import subprocess
import os

#directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter/sorting'
#os.chdir(directory)

def get_bc_democracy(bclist, bc_seq):
    """
    Function to get most common barcode sequence from collapsed barcode list.
    This is to avoid taking barcode sequences that don't represent the majority (and so don't match across samples)
    Function iterates through bowtie output for list of connected components to locate component with fewest mismatches
    Often several sequences have share the minimum number of mismatches (as they're the same sequence),
    Thus, the output returns  the lowest FASTA line number with min mismatch number

    """
    all_bc_dictlist = []
    seq_tab = pd.DataFrame(columns=['barcode', 'sum'])
    for number in bclist:
        newlist = bc_seq.loc[bc_seq['barcode'] == number, 'mismatch']
        if not newlist.isnull().all():
            newlist = newlist.str.len()
        tot = newlist.fillna(0)
        barcodedict = {'barcode': number, 'sum': sum(tot)}
        all_bc_dictlist.append(barcodedict)
        seq_tab = pd.DataFrame.from_dict(all_bc_dictlist)

    return min(seq_tab.loc[seq_tab['sum'] == min(seq_tab['sum']), 'barcode'])


def barcodecollapsing(directory, minbarcode):
    """
    Function to collapse matched barcodes from bowtie output.

    Also makes a fig for barcode counts
    Saves output as barcode and spike-in RNA barcodes separately per sample.

    Arguments:
    directory = barcode sorting folder where bowtie output is.
    minbarcode = threshold of barcode counts to call barcode real or not. Put as 0 if no thresholding
    """
    path = directory / 'sorting'
    os.chdir(path)
    for barcodefile in os.listdir(path):
        if barcodefile.startswith("out_barcode_"):
            f_size = os.path.getsize(barcodefile)
            barcodenum = barcodefile.split('out_barcode_bowtiealignment_', 1)[1]
            suffix = '.txt'
            barcodenum_nosuff = barcodenum[:-len(suffix)]
            print('Loading barcodefile. %s' %barcodefile, flush=True)
            alignedbarcode = np.loadtxt(barcodefile, dtype=int);
            if f_size == 0:
                print('%s is empty' %barcodefile)
            if alignedbarcode.ndim==1:
                print ('%s is ndims too small' %barcodefile)
            if alignedbarcode.ndim>1:
                print('Starting collapsing %s' % barcodenum, flush=True)
                G=nx.Graph()
                G.add_edges_from(alignedbarcode)
                print('Connected components %s' % barcodenum, flush=True)
                bc_seq = pd.read_csv(barcodefile[4:], delimiter = "\t", header=None)
                bc_seq = bc_seq.rename(columns={0: "barcode", 7: "mismatch"})
                barcodes_sorted= pd.DataFrame(map(lambda x: (get_bc_democracy(x, bc_seq), len(x)), nx.connected_components(G)))
                barcodes_sorted= barcodes_sorted.rename(columns={0: "barcode", 1: "frequency"})
                barcode_final = barcodes_sorted[barcodes_sorted.frequency >minbarcode]
    #plot histogram for frequency distribution
                freq = barcodes_sorted.frequency
                n, bins, patches = plt.hist(freq, color='steelblue', edgecolor='none', bins='auto', log=True)
                plt.xlabel('Copies of Barcode')
                plt.ylabel('Frequency')
                plt.title('Neuron Barcode Distribution for %s' %barcodenum_nosuff)
                figname = 'barcodeplot_%s.png' %barcodenum_nosuff
                plt.savefig(figname)

                barcodedir = 'FASTA_UMIcollapsed_%s' % barcodenum
                barcodes = np.loadtxt(barcodedir, dtype=str, delimiter = " ");
                line = (barcodes[::2])
                sequence = (barcodes[1::2])
                line = np.char.strip(line, chars ='>')
                barcode_seq = pd.DataFrame(columns = ['line', 'sequence'])
                barcode_seq['line'] = line
                barcode_seq['sequence'] = sequence
                barcode_final['sequence'] = barcode_final.line.map(barcode_seq.sequence)
                barcode_final['is_spike'] = barcode_final['sequence'].str.contains('ATCAGTCA', regex=True)
                spike =barcode_final[barcode_final['is_spike']==True]
                barcode_final = barcode_final[barcode_final['is_spike']==False]
                spikename = 'finalspikebarcodes_%s' %barcodenum
                barcodename = 'final_barcodes_%s' %barcodenum
                barcode_final.to_csv(barcodename)
                spike.to_csv(spikename)
    print('finished preprocessing barcodes')
    #may want to remove spike-in sooner and also specify by position rather than just containing string by itself?
