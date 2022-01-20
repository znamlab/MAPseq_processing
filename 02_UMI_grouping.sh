#!/bin/sh

#Script to take UMI of barcodes in each sample and groups them
cd /camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/Sequencing/Processed_data/BRAC5676.1h/trial/unzipped/barcodesplitter

#filter out reads with Ns, cut off sample barcodes need to change to for loop when have multiple samples
awk "NR%2==0" BC5.txt | grep -v N | cut -b 1-50 |sort -nr > TUR4405A1processedBC5.txt

mkdir indexes

ml Bowtie

#take each sequence and align to other sequences in the index library (same library, so all to all)
nl TUR4405A1processedBC5.txt | awk '{print ">" $1 "\n" $2}' > TUR4405A1processedBC5_fasta.txt; bowtie-build -q TUR4405A1processedBC5_fasta.txt indexes/BC5fasta; bowtie -v 3 -p 10 -f --best -a indexes/BC5fasta TUR4405A1processedBC5_fasta.txt bowtiealignment1_1.txt

#take sequence line identifiers and alignments and put in one file
awk '{print $1, $3}' bowtiealignment1_1.txt > bowtiealignment1_4.txt
