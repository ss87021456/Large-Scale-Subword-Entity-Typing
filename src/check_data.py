import pickle as pkl
import pandas as pd
import numpy as np

# Check if there's any mention appears across the sets
file = "data/smaller_preprocessed_sentence_keywords_labeled.tsv"
dataset = pd.read_csv(file, sep='\t', names=['label', 'context', 'mention'])
mentions = dataset['mention'].values[:None]
train_idx = pkl.load(open("model/train_index.pkl", 'rb'))
test_idx = pkl.load(open("model/test_index.pkl", 'rb'))
val_idx = pkl.load(open("model/validation_index.pkl", 'rb'))
del dataset

print("train instances: {0}".format(len(set(mentions[train_idx]))))
print("test  instances: {0}".format(len(set(mentions[test_idx]))))
print("val   instances: {0}".format(len(set(mentions[val_idx]))))

intersection = set(mentions[train_idx]).intersection(set(mentions[test_idx]))
print("Overlap (train & test): {0}".format(len(intersection)))
intersection = set(mentions[test_idx]).intersection(set(mentions[val_idx]))
print("Overlap (test  & val ): {0}".format(len(intersection)))
intersection = set(mentions[train_idx]).intersection(set(mentions[val_idx]))
print("Overlap (train & val ): {0}".format(len(intersection)))
