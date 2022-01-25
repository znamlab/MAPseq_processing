import subprocess
import os

verbose = 1  # 0 is silent, 1 prints progress, >2 also prints barcode splitter output
#Script to take raw data files and split by sample
os.chdir('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped')


if verbose:
    print('Split sequence and merge reads', flush=True)

with open('TUR4405A1_S1_L001_R1_001.fastq', 'r') as read1, \
    open('TUR4405A1_S1_L001_R2_001.fastq', 'r') as read2, \
    open('TUR4405A1_PE.txt', 'w') as target:
    for il, (r1, r2) in enumerate(zip(read1, read2)):
        if il % 4 == 1:
            target.write(r1.strip() + r2[:30] + '\n')

#split dataset according to inline indexes using fastx toolkit; this by default allows up to 1 missmatch. we could go higher if we want, though maybe not neccessary
if not os.path.isdir('barcodesplitter'):
    os.mkdir('barcodesplitter')
os.chdir('barcodesplitter')

# This does not work:
# subprocess.run(['ml', 'FASTX-Toolkit'])
# so the module loading must be handled before starting the python script

# barcode splitter is looking for a file with the sequence number on one line and the
# sequence on the next.
# this was done by doing:
# nl  ../TUR4405A1_PE.txt | awk '{print ">" $1 "\n" $2}'
# and pipe-ing that into stdin of barcode_splitter. That's 3 pipes and is a bit of a
# mess. Let's just make an intermediary file with python
source_file = '../TUR4405A1_PE.txt'
fasta_seq = 'barcode_splitter_input.fasta'
with open(fasta_seq, 'w') as target:
    with open(source_file, 'r') as txt_file:
        for index, line in enumerate(txt_file):
            target.write('> {0}\n{1}'.format(index + 1, line))

# now run barcode splitter on that
bcfile = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Reference_files/sample_barcodes.txt'
prefix = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped/barcodesplitter/'

if verbose:
    print('Barcode splitter', flush=True)
with open(fasta_seq, 'r') as file_input:
    out = subprocess.run(['fastx_barcode_splitter.pl', '--bcfile', bcfile, '--prefix',
                          prefix, '--eol', '--suffix', '.txt'], stdin=file_input,
                         capture_output=True)
if out.stderr:
    raise IOError('Barcode splitter raised an error:\n{0}', out.stderr)

log_file = 'barcode_splitter_log.txt'
with open(log_file, 'wb') as log:
    log.write(out.stdout)

if verbose > 1:
    print(out.stdout.decode(), flush=True)

if verbose:
    print('Sample splitting done')