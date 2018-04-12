import argparse
import json
import numpy as np
from itertools import chain
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from utils import write_to_file, readlines, generic_threading, keywords_as_labels, merge_dict
import pandas as pd
from collections import Counter

# python src/label.py data/
# python src/label.py data/ --labels=data/label.json --replace --corpus=data/smaller_preprocessed_sentence_keywords.tsv --thread=10

# python src/label.py data/ --corpus=data/smaller_preprocessed_sentence_keywords.tsv --stat


def fit_encoder(keywords_path, model=None, output=None):
    """
    Arguments:
        keywords(str): Path to directory of keywords dictionary.
        one_hot(bool): Use one-hot encoding.
        model(str): Label Encoder save filename.
        output(str): Path to the output file.
    """
    if model is None:
        model = "model/label_encoder.pkl"
    if output is None:
        output = "data/label.json"

    # Load keywords
    contents = merge_dict(keywords_path)
    # contents = json.load(open(keywords, "r"))

    # LabelEncoder
    print("Initializing LabelEncoder for encoding unique types...")
    encoder = LabelEncoder()
    # Unique the mentions
    mentions = list(contents.values())
    unique_types = list(np.unique(list(chain.from_iterable(mentions))))
    print(" - Total number of unique types: {0}".format(len(unique_types)))

    # Fit LabelEncoder
    print(" - Fitting LabelEncoder with unique types...")
    encoder.fit(unique_types)

    # Dump model to pickle file
    print(" - Saving LabelEncoder to file {0}".format(model))
    # np.save(model, encoder.classes_)
    joblib.dump(encoder, model)
    print()

    # Encode the types
    ### Consider use multi-hot encoder? ###
    print("Encoding unique types...")
    codes = encoder.transform(unique_types)

    # Save the labels and names as json
    output_dict = dict(zip(unique_types, [int(itr) for itr in codes]))
    write_to_file(output, output_dict)

def replace_labels(keywords_path, labels, output, replace, corpus, thread):
    """

    Arguments:
        keywords_path():
        output():
        replace():
        corpus():
        thread():
    """
    if output is None:
        output = corpus[:-4] + "_labeled.tsv"

    # Load lines from corpus
    raw_data = readlines(corpus, limit=None)

    # Load keywords and labels
    # Used for matching each mention and its corresponding types (text)
    mentions = merge_dict(keywords_path)
    # Used for matching each type to its corresponding labels (int)
    print("Loading labels dictionary from file: {:s}".format(labels))
    contents = json.load(open(labels, "r"))
    print(" - Total number of labels: {0}".format(len(contents)))

    # Threading
    param = (mentions, contents, "SINGLE")
    result = generic_threading(thread, raw_data, keywords_as_labels, param)

    # Write result to file
    write_to_file(output, result)

def acquire_statistic(corpus, keywords_path, output=None):
    """
    """
    if output is None:
        output = corpus[:-4] + "_stat.json"

    # Load lines from corpus
    raw_data = readlines(corpus, limit=None)
    raw_data = [[itr[:itr.find("\t")], itr[itr.find("\t") + 1:]]
                for itr in raw_data]
    df = pd.DataFrame(raw_data, columns=["CORPUS", "MENTIONS"])
    mentions = [itr.split("\t") for itr in df["MENTIONS"].as_matrix()]
    mentions = list(chain.from_iterable(mentions))
    del df
    stat = Counter(mentions)

    # Save statistics to file
    write_to_file(output, stat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("keywords_path", type=str,
                        help="Path to where mention dictionaries are saved.")
    parser.add_argument("--model", help="Output encoder name.")
    parser.add_argument("--output", help="Sentences with key words")
    #
    parser.add_argument("--stat", action="store_true",
                        help="Acquire statistic about the amount of data in mentions.")
    #
    parser.add_argument("--replace", action="store_true", help="Replace labels.")
    parser.add_argument("--labels", help="Points to data/label.json \
                        (when replacing labels).")
    parser.add_argument("--duplicate", action="store_true",
                        help="Duplicate sentences if multiple mentions are found.")
    parser.add_argument("--corpus", help="Input labeled sentences to be replaced.")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 

    args = parser.parse_args()

    if args.replace:
        replace_labels(args.keywords_path, args.labels, args.output, args.replace, args.corpus, args.thread)
    elif args.stat:
        acquire_statistic(args.corpus, args.keywords_path, args.output)
    else:
        fit_encoder(args.keywords_path, args.model, args.output)