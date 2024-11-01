'''
https://habr.com/ru/companies/sberdevices/articles/730088/
https://github.com/Den4ikAI/FRED-T5-Finetuning/blob/main/sft.py


The training data should contains source and target in each line, which should be separate by '\t'
'''
from transformers import T5ForConditionalGeneration, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset
import argparse
import os

# hf_model = "ai-forever/FRED-T5-1.7B"
hf_model = "ai-forever/FRED-T5-large"
# hf_model = 'ai-forever/ruT5-large'
# my_model = 'ai-forever/ruT5-large'
my_model = './models_bss'

class TrainerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep="\t")
        df = df.dropna()
        self.dataset = df
        self.tokenizer = GPT2Tokenizer.from_pretrained(hf_model, eos_token='</s>')
        # self.tokenizer = T5Tokenizer.from_pretrained(hf_model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source = self.dataset.iloc[idx, 0]
        target = self.dataset.iloc[idx, 1]
        input_ids = self.tokenizer.encode(source, return_tensors='pt',
                                          padding='max_length', truncation='longest_first', max_length=512)[0]
        label = self.tokenizer.encode(target, return_tensors='pt', padding='max_length',
                                      truncation='longest_first', max_length=1)[0]
        # print({'input_ids': input_ids, 'labels': label})
        return {'input_ids': input_ids, 'labels': label}


parser = argparse.ArgumentParser(description='Train docTquery on more datasets')
parser.add_argument('--pretrained_model_path', default=my_model, help='pretrained model path')
parser.add_argument('--train_data_path', default="queries_answers_dataset_train.txt", required=False, help='training data path')
parser.add_argument('--output_path', default="models_bss_fred", required=False, help='output directory path')
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--weight_decay', default=5e-5, type=float)
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('--gra_acc_steps', default=8, type=int)
args = parser.parse_args()

model = T5ForConditionalGeneration.from_pretrained(hf_model)
train_dataset = TrainerDataset(args.train_data_path)

training_args = TrainingArguments(
    output_dir=args.output_path,
    overwrite_output_dir=True,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    save_steps=500,
    learning_rate=args.lr,
    gradient_accumulation_steps=args.gra_acc_steps,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
trainer.save_model()