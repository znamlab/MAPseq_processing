import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import (draw, DiGraph, Graph)
import os

os.chdir('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter')
directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter'
if not os.path.isdir('sorting'):
    os.mkdir('sorting')

for barcodefile in os.listdir(directory):
    if barcodefile.startswith("out"):
        alignedUMI = np.loadtxt(barcodefile, dtype=int);
        G=nx.Graph()
        G.add_edges_from(alignedUMI)
        UMIs =(sorted(nx.connected_components(G), key = len, reverse=True))
        UMI_sorted = pd.DataFrame(columns=['line', 'UMI_frequency'])
        for i in UMIs:
            UMI_sorted = UMI_sorted.append({'line': (min(i)), 'UMI_frequency': (len(i))},ignore_index=True)

#if required specify a minimum UMI count threshold for minUMI
        minUMI = 0
        UMI_final = UMI_sorted[UMI_sorted.UMI_frequency >minUMI]

#save figure for UMI count make this to be saved
        barcodenum = barcodefile.split('out_bowtiealignment_', 1)[1]
        suffix = '.txt'
        barcodenum_nosuff = barcodenum[:-len(suffix)]
        plt.figure()
        plt.xlabel('Sequence Rank')
        plt.ylabel('UMI counts')
        plt.title('UMI Distribution for %s') % barcodenum_nosuff
        l=  len(UMI_final)
        x = list(range(0, l))
        plt.semilogy(x, UMI_final['UMI_frequency'])
        figname = 'sorting/UMIplot_%s.png' % barcodenum_nosuff
        plt.savefig(figname)

#take actual barcode sequences and put with lines for each UMI listed
        barcodedir = 'FASTA_TUR4405A1processed_%s' % barcodenum
        barcodes = np.loadtxt(barcodedir, dtype=str, delimiter = " ");
        line = (barcodes[::2])
        sequence = (barcodes[1::2])
        line = np.char.strip(line, chars ='>')
        barcode_seq = pd.DataFrame(columns = ['line', 'sequence'])
        barcode_seq['line'] = line
        barcode_seq['sequence'] = sequence
        UMI_final['sequence'] = UMI_final.line.map(barcode_seq.sequence)
        tosave = 'sorting/UMIgroups_%s.csv' % barcodenum_nosuff
        UMI_final.to_csv(tosave)
print("finished sorting the UMIII's")
