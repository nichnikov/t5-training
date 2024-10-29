"""тестирование модели на данных из текстового файла: queries_answers_dataset_not_lem_test.txt"""

import os
import re
import torch
from transformers import (
                          T5Tokenizer,
                          T5ForConditionalGeneration)
import pandas as pd
import time

device='cuda'
hf_model = 'ai-forever/ruT5-large'
tokenizer = T5Tokenizer.from_pretrained(hf_model)
model = T5ForConditionalGeneration.from_pretrained(os.path.join("models_bss"))
model.to(device)

"""Загрузка тестового файла:"""
with open(os.path.join("dataset", "queries_answers_dataset_not_lem_test.txt"), "r") as f:
    texts = f.read()

texts_list = texts.split("\n")
print(texts_list[:10])

test_results = []
for num, text in enumerate(texts_list):
    try:
        start = time.time()
        text_for_predict = re.sub("\t.*", "", text)
        true_label_list = re.findall("\t.*", text)
        true_label = re.sub("\t", "", true_label_list[0])
        input_ids = tokenizer.encode(text,  return_tensors="pt").to(device)
        outputs=model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True).to(device)
        outputs_decode = tokenizer.decode(outputs[0][1:])

        outputs_logits=model.generate(input_ids, output_scores=True, return_dict_in_generate=True, eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True)
        sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
        predict_label = re.sub("</s>", "", outputs_decode)
        print(num, "/", len(texts_list), "true_label:", true_label, "predict_label:", predict_label)

        d = {"predict_label": predict_label,
             "predict_score": sigmoid_0[2].item(),
             "true_label:": true_label}
        test_results.append(d)

    except:
        print("we have problem with text: {}".format(str(text)))

test_results_df = pd.DataFrame(test_results)
print(test_results_df)

# Сравнение со старой моделью:
# test_results_df.to_csv(os.path.join("results", "validated_queries_answers_41week_not_lem_240109_1400_true.csv"), sep="\t", index=False)
test_results_df.to_csv(os.path.join("results", "test_results_not_lem_240110_1200_oldbss.csv"), sep="\t", index=False)