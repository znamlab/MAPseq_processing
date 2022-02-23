import subprocess
import os

os.chdir('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter')

directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped1/barcodesplitter'

for barcodefile in os.listdir(directory):
    if barcodefile.startswith("BC"):
        newfile = 'TUR4405A1processed_%s' % barcodefile
        with open(barcodefile, 'r') as bcreads, \
            open(newfile, 'w') as target:
            for il, sequence in enumerate(bcreads):
                if il % 2== 1 and 'N' not in sequence:
                    target.write(sequence[:50] + '\n')
print('Processed each set of sample sequences for bowtie input')
#now run bowtie to take each sequence and align to other sequences in the index library (same library, so all to all)
if not os.path.isdir('indexes'):
    os.mkdir('indexes')
#make sequences in FASTA format for bowtie alignment
for barcodefile in os.listdir(directory):
    if barcodefile.startswith("TUR4405A1processed_"):
        newfile = 'FASTA_%s' % barcodefile
        with open(barcodefile, 'r') as bcreads, \
            open(newfile, 'w') as target:
            for il, sequence in enumerate(bcreads, start=1):
                target.write('>' + str(il) + '\n' + sequence)
verbose = 1
if verbose:
    print('Bowtie alignment starting', flush=True)

#Loop through each FASTA file and perform bowtie alignment.
for barcodefile in os.listdir(directory):
    if barcodefile.startswith("FASTA_TUR4405A1processed_BC"):
        barcodenum = barcodefile.split('FASTA_TUR4405A1processed_', 1)[1]
        suffix = '.txt'
        barcodenum_nosuff = barcodenum[:-len(suffix)]
        newfile = 'bowtiealignment_%s' % barcodenum
        index = 'indexes/' + barcodenum_nosuff
        library = subprocess.run(['bowtie-build', barcodefile, index], capture_output=True)
        bowtie = subprocess.run(['bowtie', '-v', '3', '-p', '10', '-f', '--best', '-a', index, barcodefile, newfile], capture_output=True)

#take first and third fields of bowtie alignments as output for next steps
for alignment in os.listdir(directory):
    if alignment.startswith("bowtiealignment_"):
        fileoutput = 'out_%s' % alignment
        with open (alignment,'r') as f, \
            open(fileoutput, 'w') as target:
            for x in f.readlines():
                target.write(x.split('\t')[0] + ' ' + x.split('\t')[2] + '\n')
print('Finished aligning UMI's)
