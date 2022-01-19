#!/bin/sh

#unzip original datafiles
gunzip *.gz


#strip fastq files and clip sequences

awk "NR%4==2" TUR4405A1_S1_L001_R1_001.fastq > TUR4405A1_1_stripped.txt #whole read just sequence
awk "NR%4==2" TUR4405A1_S1_L001_R2_001.fastq | cut -b 1-30 > TUR4405A1_2_stripped.txt #14nt UMI + 16nt sample barcode

# rezip input files to save disk space
gzip *.fastq


#make a new file that contains only one sequence per sequenced cluster
paste -d '' TUR4405A1_1_stripped.txt TUR4405A1_2_stripped.txt > TUR4405A1_PE.txt

#remove sequences that don't contain RT barcode 5 and don't have YYGTAC at end of R1
grep -E '^.{30}[CT][CT]GTAC' TUR4405A1_PE.txt > TUR4405A1_PE_R1corr.txt
fgrep -h "AGAAGGTAAACTCCGT" TUR4405A1_PE.txt > TUR4405A1_PE_RT_anypos.txt
grep -E '^.{50}AGAAGGTAAACTCCGT' TUR4405A1_PE.txt > TUR4405A1_PE_RTBARend.txt
grep -E '^.{50}AGAAGGTAAACTCCGT' TUR4405A1_PE_R1corr.txt > TUR4405A1_PE_RTBARend_R1corr.txt

#define spike-in's
grep -E '^.{24}ATCAGTCA' TUR4405A1_PE.txt > spikeinR1.txt
grep -E '^.{50}AGAAGGTAAACTCCGT' spikeinR1.txt > TUR4405A1_spikeinR1and2.txt


#count unique neuron barcodes in R1
grep -E '^.{30}[CT][CT]GTAC' TUR4405A1_1_stripped.txt > R1.txt
sort R1.txt > R1_sorted.txt
uniq -c R1_sorted.txt > R1_unique.txt

#remove barcodes that appear less than 10x
awk '$1 > 10' R1_unique.txt > R1_unique_thresholded.txt

#remove barcodes apprearing less than 100x. NB needs to be UMI collapsed to truly locate soma values
awk '$1 > 100' R1_unique.txt > R1_unique_soma_thresholded.txt

#do the same with UMI's
sort TUR4405A1_PE_RTBARend_R1corr.txt > PE_sorted.txt
uniq -c PE_sorted.txt > PE_unique.txt
