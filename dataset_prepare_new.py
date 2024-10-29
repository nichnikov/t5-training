import os
# import argparse
# from tqdm import tqdm
from random import shuffle
import pandas as pd

# qr_ans_train_df = pd.read_csv(os.path.join("dataset", "queries_answers_lem_train.tsv"), sep="\t")
# qr_ans_test_df = pd.read_csv(os.path.join("dataset", "queries_answers_lem_test.tsv"), sep="\t")

qr_ans_train_df = pd.read_csv(os.path.join("dataset", "queries_answers_not_lem_train.tsv"), sep="\t")
qr_ans_test_df = pd.read_csv(os.path.join("dataset", "queries_answers_not_lem_test.tsv"), sep="\t")


qr_ans_dicts_train = qr_ans_train_df.to_dict(orient="records")
qr_ans_dicts_test = qr_ans_test_df.to_dict(orient="records")
shuffle(qr_ans_dicts_train)
shuffle(qr_ans_dicts_test)


with open(os.path.join("dataset", "queries_answers_dataset_not_lem_train.txt"), 'w') as fout_t5:
    for d in qr_ans_dicts_train:
        fout_t5.write(f'Query: {d["query"]} Document: {d["ShortAnswer"]} Relevant:\t{d["label"]}\n')

with open(os.path.join("dataset", "queries_answers_dataset_not_lem_test.txt"), 'w') as fout_t5:
    for d in qr_ans_dicts_test:
        fout_t5.write(f'Query: {d["query"]} Document: {d["ShortAnswer"]} Relevant:\t{d["label"]}\n')

print('Done!')