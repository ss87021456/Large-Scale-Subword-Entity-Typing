import pandas as pd
import numpy as np
import random
import argparse
import itertools
import pickle as pkl
from tqdm import tqdm
from pprint import pprint
from collections import Counter
from utils import generic_threading, readlines
from sklearn.preprocessing import MultiLabelBinarizer
import os
import csv

# python ./src/generate_index.py --input=../share_data/data_labeled_kpb.tsv --thread=20 --tag=kbp
# python ./src/generate_index.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv

np.random.seed(0)  # set numpy random seed
random.seed(0)  # set random seed


def parallel_index(thread_idx, mention_count, mentions):
    desc = "Thread {:02d}".format(thread_idx + 1)
    result = list()
    result.append(thread_idx)  # use for indicates thread order
    for key in tqdm(mention_count, position=thread_idx, desc=desc):
        index = np.where(mentions == key)[0]
        temp = [key]
        temp.append(index.tolist())
        result.append(temp)

    return result


def negative_sampling(thread_idx, pos_label, train_label, distribution,
                      label_dict, sample_amt):
    desc = "Thread {:02d}".format(thread_idx + 1)
    result = list()
    for i, pos in enumerate(tqdm(pos_label, position=thread_idx, desc=desc)):
        tmp = list()
        for j in range(sample_amt):
            n_sample = np.random.choice(a=train_label, p=distribution)
            while pos == n_sample:  # pos conflict with neg
                n_sample = np.random.choice(a=train_label, p=distribution)
            np.array(tmp.append(label_dict[n_sample]))
        result.append(tmp)
    return result


def run(model_dir,
        input,
        test_size,
        n_thread=20,
        tag=None,
        text_only=False,
        neg_sample=False,
        sample=10):
    postfix = ("_" + tag) if tag is not None else ""
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Create directory to store model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading dataset..")
    dataset = pd.read_csv(
        input,
        sep='\t',
        names=['label', 'context', 'mention', 'begin', 'end', 'description'],
        dtype=str,
        quoting=csv.QUOTE_NONE,
        nrows=None)
    dataset['mention'] = dataset['mention'].astype(str)
    mentions = dataset['mention'].values
    labels = dataset['label'].values

    # use for spliting data with mention specific
    # np.random.shuffle(mentions)
    for idx, itr in enumerate(mentions):
        if type(itr) == float or type(itr) == int:
            print(idx, itr, dataset['label'][idx], dataset['context'][idx])
    print("{0} unique mentions...".format(len(set(mentions))))
    unique, counts = np.unique(mentions, return_counts=True)
    mention_count = dict(zip(unique, counts))

    print("Processing mention_index...")
    param = (mentions, )
    key_list = list(mention_count.keys())
    # [['mention1',[indices]],['mention2',[indices]],...]
    mention_index = generic_threading(n_thread, key_list, parallel_index, param)
    mention = []
    indices = []
    order = []

    for mention_pair_thread in mention_index:
        order.append(mention_pair_thread[0])

    # print(order)
    order_idx = sorted(range(len(order)), key=lambda k: order[k])
    # print(order_idx)

    ########################################
    # TO-BE-VERIFIED CODE NECESSITY
    for thread_idx in order_idx:
        for mention_pair in mention_index[thread_idx][1:]:
            mention.append(mention_pair[0])
            indices.append(mention_pair[1])
    ########################################

    mention_index = dict(zip(mention, indices))

    total_length = mentions.shape[0]
    test_len = total_length * test_size
    train_len = total_length - 2 * test_len

    train_index = list()
    test_index = list()
    validation_index = list()

    count = 0
    print("Processing training_index...")
    ########################################
    # TO-BE REVISED TO A MORE ELEGANT SPLITTING WAY
    np.random.shuffle(unique)
    for mention in tqdm(unique):
        if count < train_len:
            count += mention_count[mention]
            train_index.append(mention_index[mention])
        elif count >= train_len and count < (train_len + test_len):
            count += mention_count[mention]
            validation_index.append(mention_index[mention])
        else:
            count += mention_count[mention]
            test_index.append(mention_index[mention])
    ########################################

    # Flatten list
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

    # negative samples, kbp version only
    # Get true (positive) labels for each instance
    pos_label = labels[train_index]
    data_size = len(pos_label)
    # Calculate the density of each labels
    distribution = Counter(pos_label)
    label_idx = []
    print("Producing distribution & lable_dict...")
    for key in tqdm(distribution):
        # Normalize probabilities
        distribution[key] = distribution[key] / data_size
        # Global index (To-Be-Implemented)
        # label_idx.append(np.where(labels == key)[0][0])
        # Local index
        label_idx.append(np.where(pos_label == key)[0][0])

    train_label = list(distribution.keys())
    train_label_prob = list(distribution.values())
    label_dict = dict(zip(list(distribution.keys()), label_idx))

    param = (
        train_label,
        train_label_prob,
        label_dict,
        sample,
    )
    neg_samples = generic_threading(n_thread, pos_label, negative_sampling, param)
    neg_samples = np.array(list(itertools.chain.from_iterable(neg_samples)))

    #neg_samples = np.empty(shape=(data_size, sample), dtype=np.int)
    #for i in tqdm(range(data_size)):
    #    for j in range(sample):
    #        n_sample = np.random.choice(a=train_label, p=train_label_prob)
    #        while pos_label[i] == n_sample: # pos conflict with neg
    #            n_sample = np.random.choice(a=train_label, p=train_label_prob)
    #        neg_samples[i][j] = label_dict[n_sample]

    filename = "{:s}pos_neg_index{:s}.pkl".format(model_dir, postfix)
    pos_neg_sample = {"positive": train_index, "negative": neg_samples}
    pkl.dump(pos_neg_sample, open(filename, 'wb'))

    filename = "{:s}training_index{:s}.pkl".format(model_dir, postfix)
    pkl.dump(train_index, open(filename, 'wb'))
    filename = "{:s}validation_index{:s}.pkl".format(model_dir, postfix)
    pkl.dump(validation_index, open(filename, 'wb'))
    filename = "{:s}testing_index{:s}.pkl".format(model_dir, postfix)
    pkl.dump(test_index, open(filename, 'wb'))

    print("Writing new_test_mention_list{:s}..".format(postfix))
    X_test_mention = mentions[test_index]
    X_train_mention = mentions[train_index]
    X_validation_mention = mentions[validation_index]

    print("{0} train unique mentions...".format(len(set(X_train_mention))))
    print("{0} validation unique mentions...".format(
        len(set(X_validation_mention))))
    print("{0} test unique mentions...".format(len(set(X_test_mention))))

    filename = model_dir + "test_mention_list{:s}.txt".format(postfix)
    with open(filename, "w") as f:
        for mention in X_test_mention:
            f.write(mention + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        nargs='?',
        type=str,
        default="model/",
        help="Directory to store models. [Default: \"model/\"]")
    parser.add_argument("--input", help="Input dataset filename.")
    parser.add_argument(
        "--text_only", action="store_true", help="Input dataset filename.")
    parser.add_argument(
        "--thread", type=int, default=20, help="Number of threads to run.")
    parser.add_argument(
        "--test_size",
        nargs='?',
        const=0.1,
        type=float,
        default=0.1,
        help="Specify the portion of the testing data to be split.\
                        [Default: 10\% of the entire dataset]")
    parser.add_argument(
        "--neg_sample",
        action="store_true",
        help="Perform negative samples for zero-shot learning.")
    parser.add_argument(
        "--sample", type=int, default=10, help="Number of negative samples.")
    parser.add_argument("--tag", type=str, help="Make tags on the files.")
    args = parser.parse_args()

    run(args.model_dir, args.input, args.test_size, args.thread, args.tag,
        args.text_only, args.neg_sample, args.sample)
