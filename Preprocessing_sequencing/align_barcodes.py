import subprocess
import os
import pandas as pd

#directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter/sorting'
#os.chdir(directory)
#if not os.path.isdir('indexes'):
#    os.mkdir('indexes')

def collapsedUMIbarcode(directory):
    """
    Function to take line numbers for collapsed UMIs and generate a FASTA format for sequences
    """
    path = directory / 'sorting'
    os.chdir(path)
    if not os.path.isdir('indexes'):
        os.mkdir('indexes')
    for barcodefile in os.listdir(path):
        if barcodefile.startswith("UMIgroups_"):
            UMI_final = pd.read_csv(barcodefile)
            sequences = UMI_final['sequence']
            barcodenum = barcodefile.split('UMIgroups_', 1)[1]
            suffix = '.csv'
            barcodenum_nosuff = barcodenum[:-len(suffix)]
            newfile = 'FASTA_UMIcollapsed_%s.txt' % barcodenum_nosuff
            with open(newfile, 'w') as target:
                for il, sequence in enumerate(sequences, start=1):
                    target.write('>' + str(il) + '\n' + str(sequence)[:30] + '\n')


#Loop through each FASTA file and perform bowtie alignment.
def bowtiebarcodes(directory):
    """
    Function to loop through each FASTA file and perform bowtie alignment
    """
    path = directory / 'sorting'
    verbose = 1
    if verbose:
        print('Bowtie starting', flush=True)
    for barcodefile in os.listdir(path):
        if barcodefile.startswith("FASTA_UMIcollapsed_"):
            barcodenum = barcodefile.split('FASTA_UMIcollapsed_', 1)[1]
            suffix = '.txt'
            barcodenum_nosuff = barcodenum[:-len(suffix)]
            newfile = 'barcode_bowtiealignment_%s' % barcodenum
            index = 'indexes/' + barcodenum_nosuff
            library = subprocess.run(['bowtie-build', barcodefile, index], capture_output=True)
            bowtie = subprocess.run(['bowtie', '-v', '3', '-p', '10', '-f', '--best', '-a', index, barcodefile, newfile], capture_output=True)

    #take first and third fields of bowtie alignments as output for next steps
    for alignment in os.listdir(path):
        if alignment.startswith("barcode_bowtiealignment_"):
            fileoutput = 'out_%s' % alignment
            with open (alignment,'r') as f, \
                open(fileoutput, 'w') as target:
                for x in f.readlines():
                    target.write(x.split('\t')[0] + ' ' + x.split('\t')[2] + '\n')
    print('Finished aligning barcodes')
