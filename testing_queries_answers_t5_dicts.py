import os
import re
import torch
from transformers import (
                          T5Tokenizer,
                          T5ForConditionalGeneration)
import pandas as pd
import time

device='cuda'
# device='cpu'
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
hf_model = 'ai-forever/ruT5-large'

"""Тестовый файл:"""
# queries_answers_df = pd.read_csv(os.path.join("dataset", "queries_answers_not_lem_test.tsv"), sep="\t")

"""# Сравнение со старой моделью: """
# queries_answers_df = pd.read_csv(os.path.join("data", "validate_queries_41week_mouses_compare_lm.csv"), sep="\t")

"""# Тестовая выборка данных: """
# queries_answers_df = pd.read_csv(os.path.join("data", "all_sys_qrs_with_answers_lm_qrs.csv"), sep="\t")
queries_answers_df = pd.read_csv(os.path.join("results", "validated_all_sys_qrs_with_answers.csv"), sep="\t")

queries_answers_dicts = queries_answers_df.to_dict(orient="records")
print(queries_answers_df)
print(queries_answers_dicts[:10])

tokenizer = T5Tokenizer.from_pretrained(hf_model)
# model = T5ForConditionalGeneration.from_pretrained(os.path.join("models_all_sys", "checkpoint-5000"))
model = T5ForConditionalGeneration.from_pretrained(os.path.join("models_bss"))
model.to(device)

test_results = []
print("All examples:", len(queries_answers_dicts))
for num, d in enumerate(queries_answers_dicts):
    try:
        start = time.time() # fa_text
        # text = "Query: " + d["lm_query"] + " Document: " + d["FastAnswerText"] + " Relevant: " # Old model BSS
        text = "Query: " + d["lm_query"] + " Document: " + d["fa_text"] + " Relevant: " # Old model BSS
        input_ids = tokenizer.encode(text,  return_tensors="pt").to(device)
        outputs=model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True).to(device)
        outputs_decode = tokenizer.decode(outputs[0][1:])

        outputs_logits=model.generate(input_ids, output_scores=True, return_dict_in_generate=True, eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True)
        sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
        print(num, "/", len(queries_answers_dicts), "outputs:", outputs, "outputs decode:", outputs_decode, "sigmoid:", sigmoid_0[2], "time:", time.time() - start)
        d["BssMouseRelevant"] = re.sub("</s>", "", outputs_decode)
        d["BssMouseScore"] = sigmoid_0[2].item()
        test_results.append(d)
    except:
        print("we have problem with text: {}".format(str(text)))

test_results_df = pd.DataFrame(test_results)
print(test_results_df)

""" # Сравнение со старой моделью: """
test_results_df.to_csv(os.path.join("results", "validated_all_sys_qrs_with_answers.csv"), sep="\t", index=False)