import pandas as pd
from pprint import pprint
import numpy as np
import argparse
import itertools
from tqdm import tqdm
import pickle as pkl
from utils import generic_threading
from sklearn.preprocessing import MultiLabelBinarizer
import os

# python ./src/generate_index.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv 

np.random.seed(0) # set random seed

def parallel_index(thread_idx, mention_count, mentions):
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    for key in tqdm(mention_count, position=thread_idx, desc=desc):
        index = np.where(mentions == key)[0]
        temp = [key]
        temp.append(index.tolist())
        result.append(temp)

    return result

def run(model_dir, input, test_size):
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Create directory to store model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading dataset..")
    dataset = pd.read_csv(input, sep='\t', names=['label','context','mention'])

    mentions = dataset['mention'].values[:None]

    
    # use for spliting data with mention specific 
    np.random.shuffle(mentions)
    print("{0} unique mentions...".format(len(set(mentions))))
    unique, counts = np.unique(mentions, return_counts=True)
    mention_count = dict(zip(unique, counts))

    print("processing mention_index...")
    param = (mentions, )
    key_list = list(mention_count.keys())
    # [['mention1',[idxes]],['mention2',[idxes]],...]
    mention_index = generic_threading(20, key_list, parallel_index, param) 
    mention = []
    indices = []

    for metion_pair_thread in mention_index:
        for metion_pair in metion_pair_thread:
            mention.append(metion_pair[0])
            indices.append(metion_pair[1])

    mention_index = dict(zip(mention, indices))

    total_length = mentions.shape[0]
    test_len     = total_length * test_size
    train_len    = total_length - test_len
    train_index  = []
    count = 0
    print("processing training_index...")
    print("training size: {0}, testing size: {1}, total size: {2}".format(train_len, test_len, total_length))
    for mention in tqdm(key_list):
        if count < train_len:
            count += mention_count[mention]
            train_index.append(mention_index[mention])

    # flatten list
    print("flatten train_index...")
    dataset_index = set([i for i in range(total_length)])
    train_index = set(list(itertools.chain.from_iterable(train_index)))
    test_index = dataset_index - train_index  # use set property to extract index of testing
    print("train size:",len(train_index))
    train_index = np.array(list(train_index)) # transfer back to numpy array for further index
    test_index = np.array(list(test_index))   # transfer back to numpy array for further index

    # shuffle the index
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)
    #print("train_index:",train_index)
    #print("test_index:",test_index)
    pkl.dump(train_index, open(model_dir + "train_index.pkl", 'wb'))
    pkl.dump(test_index, open(model_dir + "test_index.pkl", 'wb'))
    
    
    print("Loading pkl...")
    train_index = pkl.load(open(model_dir + "train_index.pkl", 'rb'))
    test_index = pkl.load(open(model_dir + "test_index.pkl", 'rb'))

    print("Writing new_test_mention_list..")
    X_test_mention = mentions[test_index]
    X_train_mention = mentions[train_index]

    print("{0} test unique mentions...".format(len(set(X_test_mention))))
    print("{0} train unique mentions...".format(len(set(X_train_mention))))

    with open(model_dir + "test_mention_list.txt", "w") as f:
        for mention in X_test_mention:
            f.write(mention + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", nargs='?', type=str, default="model/", 
                        help="Directory to store models. [Default: \"model/\"]")
    parser.add_argument("--input", help="Input dataset filename.")
    parser.add_argument("--test_size", nargs='?', const=0.1, type=float, default=0.1,
                        help="Specify the portion of the testing data to be split.\
                        [Default: 10\% of the entire dataset]")
    args = parser.parse_args()

    run(args.model_dir, args.input, args.test_size)
