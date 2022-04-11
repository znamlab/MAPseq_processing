import subprocess
import os


directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter'

def groupingUMI_restructure(directory):

    """
    Function to take sequence reads that have been split into sample barcode files.
    (i) Restructure the barcode splitter output so take every other line, cut to 50nt, take out N's
    (ii) Make back into FASTA format
    """

    os.chdir(directory)
    if not os.path.isdir('indexes'):
        os.mkdir('indexes')

    for barcodefile in os.listdir(directory):
        linenum = 1
        if barcodefile.startswith("BC"):
            newfile = 'processed_%s' % barcodefile
            with open(barcodefile, 'r') as bcreads, \
                open(newfile, 'w') as target:
                for il, sequence in enumerate(bcreads):
                    if il % 2== 1 and 'N' not in sequence:
                        target.write('>' + str(linenum) + '\n' + sequence[:50] + '\n')
                        linenum += 1
    print('Processed each set of sample sequences for bowtie input')

def UMI_bowtie(directory):

    """
    Function to run bowtie, taking each sequence in sample barcode library and align to other sequences in
    the index library (same library, so all to all) by feeding into and running bowtie alignment to match UMI's

    Args
    directory = directory where sample splitting occurred
    """
    verbose = 1
    if verbose:
        print('Bowtie alignment starting', flush=True)

    for barcodefile in os.listdir(directory):
        if barcodefile.startswith("processed_"):
            barcodenum = barcodefile.split('processed_', 1)[1]
            suffix = '.txt'
            barcodenum_nosuff = barcodenum[:-len(suffix)]
            newfile = 'bowtiealignment_%s' % barcodenum
            index = 'indexes/' + barcodenum_nosuff
            library = subprocess.run(['bowtie-build', barcodefile, index], capture_output=True)
            bowtie = subprocess.run(['bowtie', '-v', '3', '-p', '10', '-f', '--best', '-a', index, barcodefile, newfile], capture_output=True)
            print('bowtie alignment finished')

def restructurebowtie(directory):
    """
    Function to take first and third fields of bowtie alignments as output for next steps
    """
    for alignment in os.listdir(directory):
        if alignment.startswith("bowtiealignment_"):
            fileoutput = 'out_%s' % alignment
            with open (alignment,'r') as f, \
                open(fileoutput, 'w') as target:
                for x in f.readlines():
                    target.write(x.split('\t')[0] + ' ' + x.split('\t')[2] + '\n')
    print('Finished aligning UMI's)
