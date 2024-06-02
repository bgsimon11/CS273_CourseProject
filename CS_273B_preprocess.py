import pandas as pd
import numpy as np
import requests
import subprocess
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import DataFrame
import pyreadr

#import data and format frame
df = pd.read_csv('/Users/Delaney_Smith_1/Desktop/CS273B/GSE152431_rna_raw_counts.csv')
df.set_index('sampleName', inplace=True)

#column-wise (patient-level) normalization to TPM
normalized_df = df.div(df.sum(axis=0), axis=1) * 1000000

#row-wise (gene-level) min-max normalization 
mm_df = normalized_df
#mm_df = normalized_df.apply(lambda row: (row - row.min()) / (row.max() - row.min()), axis=1)

#save formatted dataframe
#mm_df.to_csv('normalized_GSE152431_rna_raw_counts.csv')

#check for protein coding genes using the fasta file as a df
def parse_fasta_header(header):
    parts = header.split('|')
    return parts

def fasta_headers_to_df(file_path):
    headers = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(">"):
                headers.append(parse_fasta_header(line.strip()))
    df = pd.DataFrame(headers, columns=['ENST_id', 'ENSG_id', 'OTH_id', 'OTH2_id', 'Name', 'Gene', 'Num', 'UTR5', 'CDS', 'UTR3', ' '])
    return df

#Filter
file_path = 'gencode.v38.pc_transcripts.fa'
fasta_df = fasta_headers_to_df(file_path)
pc_genes = fasta_df['Gene'].tolist()
filtered =  mm_df.loc[mm_df.index.isin(pc_genes)]

#Save again ith only protein coding genes
#filtered.to_csv('normalized_protein_coding_GSE152431_rna_raw_counts.csv')

#Load Other.Rdata object to see gene order used to train
result = pyreadr.read_r('/Users/Delaney_Smith_1/Desktop/CS273B/Ensembl_ID.RData')
IDs = result['Ensembl_ID']
e_id = IDs.values.tolist()
e_id = [item for sublist in e_id for item in sublist]

#Add ensemble ID to our gene df
fasta_df['ENSG_id'] = fasta_df['ENSG_id'].str.split('.').str[0]
filtered = filtered.reset_index()
result = filtered.join(fasta_df.set_index('Gene')['ENSG_id'], on='sampleName')
result = result.drop_duplicates(subset='sampleName')

#Sort our genes
id_index_map = {id: index for index, id in enumerate(e_id)}
result['id_index'] = result['ENSG_id'].map(id_index_map)
result = result.sort_values(by='id_index')
result.drop('id_index', axis=1, inplace=True)

#Fill missing genes with zeros
missing_ids = list(set(e_id) - set(result['ENSG_id']))
missing_data = pd.DataFrame(0, index=missing_ids, columns=result.columns)
result.set_index('ENSG_id', inplace=True)
result = pd.concat([result, missing_data])
result.drop(columns=['ENSG_id'], inplace=True)
result =  result.loc[result.index.isin(e_id)]
result.rename(columns={'sampleName': 'Gene'}, inplace=True)

#order the patients so first 19797 rows are pre and second set are post treatment
split = pd.read_csv('/Users/Delaney_Smith_1/Desktop/Train_Val_split.csv')
headers = pd.MultiIndex.from_arrays([split.iloc[0], split.iloc[1]])
split = split.drop([0, 0])
split.columns = headers

#format the train val split df
multi_index = pd.MultiIndex.from_tuples([('train', 'Pre-Treatment'), ('train', 'Post-Treatment'), ('validation', 'Pre-Treatment'), ('validation', 'Post-Treatment')])
split.drop(split.columns[2], axis=1, inplace=True)
split.columns = multi_index
#print(split)

#make into dictionary
samples = []
#for index, row in split.iterrows():
#    samples.append((row.iloc[0], row.iloc[1]))

for index, row in split.iterrows():
    samples.append((row.iloc[2], row.iloc[3]))

samples = samples[:-26]
#print(samples)

#reshape data
pre = []
post = []
for tuple in samples:
    string = tuple[0]
    pre.append(string[:-4])
    string = tuple[1]
    post.append(string[:-4])

top = result[pre]
bottom = result[post]

#first 35 (train), 9 (val) columns are pre-treatment, and second set are post-treatment
collapsed_df = pd.concat([top, bottom], axis=1)
print(collapsed_df)

#save train and val seperately
collapsed_df.to_csv('FINAL_val_normalized_protein_coding_GSE152431_rna_raw_counts.csv')



