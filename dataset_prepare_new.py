import os
# import argparse
# from tqdm import tqdm
from random import shuffle
import pandas as pd

"""Подготовка данных:"""
df_suited = pd.read_csv(os.path.join(os.getcwd(), "data", "bss_2024.csv"), sep="\t")
df_unsuited = pd.read_csv(os.path.join(os.getcwd(), "data", "bss_2024_unsuited.csv"), sep="\t")
df_suited["label"] = "Правда"
df_unsuited["label"] = "Ложь"

dataset_df = pd.concat((df_suited[["ClearQuery", "ClearAnswer", "label"]], df_unsuited[["ClearQuery", "ClearAnswer", "label"]]))
shuffle_ds_df = dataset_df.sample(frac=1)

print(shuffle_ds_df)

test_quantity = len(shuffle_ds_df)//9
qr_ans_train_df = shuffle_ds_df[:-test_quantity]
qr_ans_test_df = shuffle_ds_df[-test_quantity:]


qr_ans_dicts_train = qr_ans_train_df.to_dict(orient="records")
qr_ans_dicts_test = qr_ans_test_df.to_dict(orient="records")
shuffle(qr_ans_dicts_train)
shuffle(qr_ans_dicts_test)


with open(os.path.join("dataset", "queries_answers_dataset_train_bss.txt"), 'w') as fout_t5:
    for d in qr_ans_dicts_train:
        fout_t5.write(f'Query: {d["ClearQuery"]} Document: {d["ClearAnswer"]} Relevant:\t{d["label"]}\n')

with open(os.path.join("dataset", "queries_answers_dataset_test_bss.txt"), 'w') as fout_t5:
    for d in qr_ans_dicts_test:
        fout_t5.write(f'Query: {d["ClearQuery"]} Document: {d["ClearAnswer"]} Relevant:\t{d["label"]}\n')

print('Done!')