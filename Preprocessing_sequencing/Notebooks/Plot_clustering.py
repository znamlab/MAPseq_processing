import os
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection
import sys
from sklearn.preprocessing import normalize
sys.setrecursionlimit(1000000)

directory = '/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/Sequencing/Processed_data/final_counts'
os.chdir(directory)

barcodes_across_sample = pd.read_pickle("barcodes_across_sample.pkl")

#normalise data to beta-actin levels 
qPCR = pd.read_csv('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq/A1_MAPseq/FIAA32.6a/qPCR/qPCR_FIAA326a.csv') 
qPCR['BactNormFactor'] = 1/np.power(2,(-(qPCR['B-actin Ct']-(max(qPCR['B-actin Ct'])))))
barcodes_norm = pd.DataFrame()
for column in barcodes_across_sample:
    NormFactor=qPCR.loc[qPCR['RT primer'] == column, 'BactNormFactor'].iloc[0]
    barcodes_norm[column]=barcodes_across_sample[column]*NormFactor
    
 #remove barcodes that are only seen in one sample   
barcodes_norm['samplesnotin'] =0
for index, row in barcodes_norm.iterrows():
    barcodes_norm['samplesnotin'].iloc[index]=(row.isna().sum())
barcodes_norm = barcodes_norm[barcodes_norm['samplesnotin']<90]
barcodes_norm = barcodes_norm.drop('samplesnotin', axis=1)

#remove NaN
barcodes_norm = barcodes_norm.fillna(0)

#remove source samples and OB
barcodes_norm = barcodes_norm.drop([40, 41, 42, 43, 49, 50, 51, 52, 1, 2, 3, 4, 5], axis=1)
barcodes_norm = barcodes_norm.loc[~(barcodes_norm==0).all(axis=1)] #remove barcodes that aren't in these samples
barcodes_norm = barcodes_norm.loc[:, (barcodes_norm!=0).any(axis=0)] #remove samples that don't have barcodes using this threshold

#HVAonly = barcodes_norm[[25, 26, 30, 31, 32, 37, 39, 44, 45, 47, 48, 58, 59, 60, 61, 62, 63, 69, 72, 73, 74, 75, 76, 80, 82, 83, 84]]
#HVAonly = HVAonly.loc[~(HVAonly==0).all(axis=1)] #remove barcodes that aren't in these samples
#HVAonly = HVAonly.loc[:, (HVAonly!=0).any(axis=0)] #remove samples that don't have barcodes using this threshold
#print(HVAonly)
#perform hierarchial clustering
#sb.clustermap(HVAonly, metric='euclidean', standard_scale=0, cmap="Blues")
#plt.savefig('heatmap_barcodes_HVA.jpg')
sb.clustermap(barcodes_norm, metric='euclidean', standard_scale=0, cmap="Blues", figsize=(60, 10))
plt.savefig('all_barcodes_nosource.jpg')
