import pandas as pd
from transformers import AutoTokenizer

path='data/jvc-20k.csv'
dataset = pd.read_csv(path)

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_training_corpus(dataset):
    for i in range(0, len(dataset["data"]), 1000):
        yield dataset["data"][i : i + 1000]

training_corpus = get_training_corpus(dataset)

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokenizer.save_pretrained("pretrained/jvc-tokenizer")