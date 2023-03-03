import os
import numpy as np
import pandas as pd
from umi_tools import UMIClusterer
from datetime import datetime
import subprocess, shlex

def getCorrectUMIs(barcodefile, directory, bigonesorting_dir, homopolymerthresh=5):
    
    """
    Function to run UMI-tools to correct PCR and sequencing errors, then take UMI counts for each barcode.

    Args
    directory = directory where sample splitting occurred
    homopolymerthres = we use a cut of for homopolymeric repeats (a common sequencing error). default=5
    """
    os.chdir(directory)
    clusterer = UMIClusterer(cluster_method="directional")
    hopolA = 'A'*homopolymerthresh
    hopolT = 'T'*homopolymerthresh
    hopolC = 'C'*homopolymerthresh
    hopolG = 'G'*homopolymerthresh
    suffix = '.txt'
    barcodefile = barcodefile.strip()
    barcodenum = barcodefile[:-len(suffix)]
    print('Starting UMI collapsing for %s' %barcodenum)
    tstart = datetime.now()
    raw_bc = pd.read_csv(barcodefile, delimiter = "\t", skiprows=lambda x: (x != 0) and not x % 2)
    barcode_tab = pd.DataFrame()
    #remove reads containing 'N'
    QCreads = raw_bc[~raw_bc.iloc[:,0].str.contains('N')]
    #remove homopolymers with [N]5 repeats
    QCreads = QCreads[~QCreads.iloc[:,0].str.contains(hopolA)]
    QCreads = QCreads[~QCreads.iloc[:,0].str.contains(hopolT)]
    QCreads = QCreads[~QCreads.iloc[:,0].str.contains(hopolC)]
    barcode_tab['full_read'] = QCreads[~QCreads.iloc[:,0].str.contains(hopolG)]
    barcode_tab['UMI_neuron_bc'] = barcode_tab['full_read'].str[32:46]
    size = os.stat(barcodefile).st_size
    if size > 1000000000: #if the size of the barcode table is huge, we'll split it into smaller chunks for processing, then cluster together again at end
        numberdivisions = round(barcode_tab.shape[0]/2000000)
        sequences = barcode_tab
        num=0
        for i in range(numberdivisions):
            if num != numberdivisions-1:
                df_short = sequences.iloc[:2000000,:]
                sequences= sequences.iloc[2000000:,:]
            else:
                df_short = sequences
            newtmpfile = 'temp/bigones/UMIintermediate_%s_%s.csv' % (barcodenum, num+1)
            df_short.to_csv(newtmpfile)
            num= num+1
        newdir =bigonesorting_dir
        UMIprocesstheminlots(newdir=newdir, barcodenum=barcodenum, numberdivisions=numberdivisions)                
        print ('split %s into %s repetitions' % (barcodenum, numberdivisions)) 
    else:
        barcode_tab['bytUMIneuron'] = [x.encode() for x in barcode_tab.UMI_neuron_bc]
        UMIcounts = barcode_tab['bytUMIneuron'].value_counts()
        UMIdict = UMIcounts.to_dict()
        clustered_umis = clusterer(UMIdict, threshold=1)
        mydict = {}
        for x in clustered_umis:
            correct = x[0]          # correct (more frequent) UMI
            if (len(x) > 1):
                for each in x[1:]:
                    mydict[each] = correct
            mydict[correct] = correct
        barcode_tab['corrected_umi'] = [mydict[x] for x in barcode_tab.bytUMIneuron]
        barcode_tab['corrected_umi'] = [x.decode() for x in barcode_tab.corrected_umi]
        corrected = sum(barcode_tab.UMI_neuron_bc != barcode_tab.corrected_umi)
        total = len(barcode_tab)
        correctedUMI = barcode_tab['corrected_umi']
        newfile = 'temp/UMIs_corrected_%s.csv' % barcodenum
        correctedUMI.to_csv(newfile)
        print('Corrected %s out of  %s UMIs for %s' % (corrected, total, barcodefile))
    tend = datetime.now()
    print('That took %s' % (tend - tstart), flush=True)
    
def UMIprocesstheminlots(newdir, barcodenum, numberdivisions):
     
    """
    Function to send a load of slurm jobs for bigger files to process in chunks of 5mil reads, so it won't take a lifetime to process.

    Args
    barcodenum = RT sample barcode identifier
    numberdivisions = the number of chunks you have for each bigger file
    """
    os.chdir(newdir)
    script_path = '/camp/home/turnerb/home/users/turnerb/code/MAPseq_processing/Preprocessing_sequencing/Bash_scripts/process_big_onesUMI.sh'
    for x in range(numberdivisions):
        toread = 'UMIintermediate_%s_%s.csv' % (barcodenum, x+1)
        command = f"sbatch {script_path} {toread}"
        print(command)
        subprocess.Popen(shlex.split(command))
        
       
def UMIprocessthechunks(directory, toread):
    """
    Function to process the chunks of 5mil reads from bigger files, to correct sequencing errors, so it won't take a lifetime to process.

    Args
    toread = split barcode csv file to read
    directory = where files are
    """
    os.chdir(directory)
    barcode_tab = pd.read_csv(toread)
    suffix = '.csv'
    barcodenum = toread[:-len(suffix)]
    print('Starting UMI collapsing for %s' %barcodenum)
    tstart = datetime.now()
    barcode_tab['bytUMIneuron'] = [x.encode() for x in barcode_tab.UMI_neuron_bc]
    UMIcounts = barcode_tab['bytUMIneuron'].value_counts()
    UMIdict = UMIcounts.to_dict()
    clusterer = UMIClusterer(cluster_method="directional")
    clustered_umis = clusterer(UMIdict, threshold=1)
    mydict = {}
    for x in clustered_umis:
        correct = x[0]          # correct (more frequent) UMI
        if (len(x) > 1):
            for each in x[1:]:
                mydict[each] = correct
        mydict[correct] = correct
    barcode_tab['corrected_umi'] = [mydict[x] for x in barcode_tab.bytUMIneuron]
    barcode_tab['corrected_umi'] = [x.decode() for x in barcode_tab.corrected_umi]
    corrected = sum(barcode_tab.UMI_neuron_bc != barcode_tab.corrected_umi)
    total = len(barcode_tab)
    correctedUMI = barcode_tab['corrected_umi']
    newfile = 'UMIs_corrected_%s.csv' % barcodenum
    correctedUMI.to_csv(newfile)
    tend = datetime.now()
    print('Corrected %s out of  %s neuron barcode counts for %s' % (corrected, total, barcodenum))
    print('That took %s' % (tend - tstart), flush=True)        