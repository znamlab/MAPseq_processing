#!/bin/sh

#Script to take raw data files and split by sample

#unzip original datafiles
gunzip *.gz


#strip fastq files and clip sequences

awk "NR%4==2" TUR4405A1_S1_L001_R1_001.fastq > TUR4405A1_1_stripped.txt #whole read just sequence
awk "NR%4==2" TUR4405A1_S1_L001_R2_001.fastq | cut -b 1-30 > TUR4405A1_2_stripped.txt #14nt UMI + 16nt sample barcode

# rezip input files to save disk space
gzip *.fastq


#make a new file that contains only one sequence per sequenced cluster
paste -d '' TUR4405A1_1_stripped.txt TUR4405A1_2_stripped.txt > TUR4405A1_PE.txt

#split dataset according to inline indexes using fastx toolkit; this by default allows up to 1 missmatch. we could go higher if we want, though maybe not neccessary
mkdir barcodesplitter
cd barcodesplitter

ml FASTX-Toolkit
nl /camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped/TUR4405A1_PE.txt |awk '{print ">" $1 "\n" $2}'|fastx_barcode_splitter.pl --bcfile /camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Reference_files/sample_barcodes.txt --prefix /camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped/barcodesplitter/ --eol
