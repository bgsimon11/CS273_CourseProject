import pandas as pd
import numpy as np

#import data and format frame
df = pd.read_csv('/Users/Delaney_Smith_1/Desktop/GSE152431_rna_raw_counts.csv')
df.set_index('sampleName', inplace=True)
#print(df.head())

#column-wise (patient-level) normalization to TPM
normalized_df = df.div(df.sum(axis=0), axis=1) * 1000000
#print(normalized_df.head())

#row-wise (gene-level) min-max normalization 
mm_df = normalized_df.apply(lambda row: (row - row.min()) / (row.max() - row.min()), axis=1)
#print(mm_df.head())

#save formatted dataframe
mm_df.to_csv('normalized_GSE152431_rna_raw_counts.csv')