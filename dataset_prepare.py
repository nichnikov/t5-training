import os
import argparse
from tqdm import tqdm
from random import shuffle
import pandas as pd

qr_ans_df = pd.read_csv(os.path.join("dataset", "queries_answers_lem_train.tsv"), sep="\t")
qr_ans_0_df = qr_ans_df[qr_ans_df["label"] == 0]
qr_ans_1_df = qr_ans_df[qr_ans_df["label"] == 1]

qr_ans_0_df_ =  qr_ans_0_df.sample(n=qr_ans_1_df.shape[0])
qr_ans_dicts = pd.concat([qr_ans_1_df, qr_ans_0_df_], axis=0).to_dict(orient="records")
print(qr_ans_dicts[:5])

shuffle(qr_ans_dicts)

train_q = 3*(len(qr_ans_dicts)//4)

"""
This script creates monoT5 input files for training,
Each line in the monoT5 input file follows the format:
    f'Query: {query} Document: {document} Relevant:\t{label}\n')
"""

with open("queries_answers_dataset_train.txt", 'w') as fout_t5:
    for d in qr_ans_dicts[:train_q]:
        if d["label"] == 1:
            relevant = "Правда"
        else:
            relevant = "Ложь"
        fout_t5.write(f'Query: {d["query"]} Document: {d["ShortAnswer"]} Relevant:\t{relevant}\n')

with open("queries_answers_dataset_test.txt", 'w') as fout_t5:
    for d in qr_ans_dicts[train_q:]:
        if d["label"] == 1:
            relevant = "Правда"
        else:
            relevant = "Ложь"
        fout_t5.write(f'Query: {d["query"]} Document: {d["ShortAnswer"]} Relevant:\t{relevant}\n')


print('Done!')