import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import (draw, DiGraph, Graph)
import subprocess
import os

#directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter/sorting'
#os.chdir(directory)

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
            alignedbarcode = np.loadtxt(barcodefile, dtype=int);
            G=nx.Graph()
            G.add_edges_from(alignedbarcode)
            barcodes =(sorted(nx.connected_components(G), key = len, reverse=True))
            barcode_sorted = pd.DataFrame(columns=['line', 'barcode_frequency'])
            for i in barcodes:
                barcode_sorted = barcode_sorted.append({'line': (min(i)), 'barcode_frequency': (len(i))},ignore_index=True)
    #if you want a minimum threshold to barcode occurance change the number
            minbarcode = 0
            barcode_final = barcode_sorted[barcode_sorted.barcode_frequency >minbarcode]


    #plot histogram for frequency distribution
            barcodenum = barcodefile.split('out_barcode_bowtiealignment_', 1)[1]
            suffix = '.txt'
            barcodenum_nosuff = barcodenum[:-len(suffix)]
            freq = barcode_sorted.barcode_frequency
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
