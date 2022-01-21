import subprocess
import os

#Script to take raw data files and split by sample
os.chdir('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped')

#strip fastq files and clip sequences

with open('TUR4405A1_1_stripped.txt', 'w') as target:
    subprocess.run(['awk', r"NR%4==2", 'TUR4405A1_S1_L001_R1_001.fastq'],
                   stdout=target)  #whole read just sequence
with open('TUR4405A1_2_stripped.txt', 'w') as target:
    p1 = subprocess.Popen(['awk', r"NR%4==2", 'TUR4405A1_S1_L001_R2_001.fastq'],
                          stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['cut', '-b', '1-30'], stdin=p1.stdout, stdout=target)  #14nt
    p1.stdout.close()
    p2.communicate()
    # UMI + 16nt sample barcode

#make a new file that contains only one sequence per sequenced cluster
with open('TUR4405A1_PE.txt', 'w') as target:
    subprocess.run(['paste', '-d', '', 'TUR4405A1_1_stripped.txt',
                    'TUR4405A1_2_stripped.txt'], stdout=target)

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
with open(fasta_seq, 'r') as file_input:
    subprocess.run(['fastx_barcode_splitter.pl', '--bcfile', bcfile, '--prefix', prefix,
                    '--eol', '--suffix', '.txt'], stdin=file_input)