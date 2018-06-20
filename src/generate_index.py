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
    result.append(thread_idx) # use for indicates thread order
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
    mentions = dataset['mention'].values

    # use for spliting data with mention specific 
    # np.random.shuffle(mentions)
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
    order = []

    for mention_pair_thread in mention_index:
        order.append(mention_pair_thread[0])

    print(order)
    order_idx = sorted(range(len(order)), key=lambda k: order[k])
    print(order_idx)

    ########################################
    # TO-BE-VERIFIED CODE NECESSITY
    for thread_idx in order_idx:
        for mention_pair in mention_index[thread_idx][1:]: # take the thread in order
            mention.append(mention_pair[0])
            indices.append(mention_pair[1])
    ########################################

    mention_index = dict(zip(mention, indices))

    total_length = mentions.shape[0]
    test_len     = total_length * test_size
    train_len    = total_length - 2 * test_len

    train_index  = list()
    test_index = list()
    validation_index = list()

    count = 0
    print("Processing training_index...")
    print("Training size: {0}, testing size: {1}, validation size: {2}, total size: {3}".format(train_len, test_len, test_len, total_length))
    ########################################
    # TO-BE REVISED TO A MORE ELEGANT SPLITTING WAY
    np.random.shuffle(unique)
    for mention in tqdm(unique):
        if count < train_len:                                       # for training dataset
            count += mention_count[mention]
            train_index.append(mention_index[mention])
        elif count >= train_len and count < (train_len + test_len): # for validation dataset
            count += mention_count[mention]
            validation_index.append(mention_index[mention])
        else :                                                      # rest are for testing dataset
            count += mention_count[mention]
            test_index.append(mention_index[mention])
    ########################################

    # flatten list
    print("Flatten train/validation/test index...")
    train_index = list(itertools.chain.from_iterable(train_index))
    validation_index = list(itertools.chain.from_iterable(validation_index))
    test_index = list(itertools.chain.from_iterable(test_index))

    print("Number of instances in all sets:")
    print(" - Training   :", len(train_index))
    print(" - Testing    :", len(test_index))
    print(" - Validation :", len(validation_index))

    train_index = np.array(train_index)
    validation_index = np.array(validation_index)
    test_index = np.array(test_index)


    # shuffle the index
    np.random.shuffle(train_index)
    np.random.shuffle(validation_index)
    np.random.shuffle(test_index)
    pkl.dump(train_index, open(model_dir + "train_index.pkl", 'wb'))
    pkl.dump(validation_index, open(model_dir + "validation_index.pkl", 'wb'))
    pkl.dump(test_index, open(model_dir + "test_index.pkl", 'wb'))
    
    
    print("Loading pkl...")
    train_index = pkl.load(open(model_dir + "train_index.pkl", 'rb'))
    test_index = pkl.load(open(model_dir + "test_index.pkl", 'rb'))
    validation_index = pkl.load(open(model_dir + "validation_index.pkl", 'rb'))

    print("Writing new_test_mention_list..")
    X_test_mention = mentions[test_index]
    X_train_mention = mentions[train_index]
    X_validation_mention = mentions[validation_index]


    print("{0} train unique mentions...".format(len(set(X_train_mention))))
    print("{0} validation unique mentions...".format(len(set(X_validation_mention))))
    print("{0} test unique mentions...".format(len(set(X_test_mention))))



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
