import argparse
import json
import numpy as np
from itertools import chain
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from utils import write_to_file, readlines, generic_threading, keywords_as_labels, merge_dict
import pandas as pd
from collections import Counter
import csv

"""
python src/label.py ../share/data.txt --from_file --tag=kpb

python src/label.py data/ --trim
python src/label.py data/ --labels=data/label.json --replace \
--corpus=data/smaller_preprocessed_sentence_keywords.tsv --subwords=data/subwords.json --thread=10

python src/label.py data/ --labels=data/label.json --replace \
--corpus=data/smaller_preprocessed_sentence_keywords.tsv --subwords=data/subwords.json --thread=10 --limit=100

python src/label.py data/ --corpus=data/smaller_preprocessed_sentence_keywords.tsv --stat
"""

def fit_encoder(keywords_path, model=None, trim=True, from_file=False, output=None, tag=None):
    """
    Arguments:
        keywords(str): Path to directory of keywords dictionary.
        one_hot(bool): Use one-hot encoding.
        model(str): Label Encoder save filename.
        output(str): Path to the output file.
    """
    postfix = ("_" + tag) if tag is not None else ""
    if model is None:
        model = "model/label_encoder{:s}.pkl".format(postfix)
    if output is None:
        output = "data/label{:s}.json".format(postfix)
        r_output = "data/label_lookup{:s}.json".format(postfix)

    # Load keywords
    if from_file:
        contents = readlines(keywords_path, delimitor="\t")
        mentions = [itr[0] for itr in contents]
        # contents = pd.read_csv(keywords_path, sep="\t", names=['label','context','mention'], dtype={'mention': str}, quoting=csv.QUOTE_NONE)
        # mentions = contents['label'].values
    else:
        # contents = json.load(open(keywords, "r"))
        contents = merge_dict(keywords_path, trim=trim)
        # Unique the mentions
        mentions = list(contents.values())

    # LabelEncoder
    print("Initializing LabelEncoder for encoding unique types...")
    encoder = LabelEncoder()

    unique_types = list(np.unique(mentions if from_file else list(chain.from_iterable(mentions))))
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
    # Reversed lookup table
    output_dict = dict(zip([int(itr) for itr in codes], unique_types))
    write_to_file(r_output, output_dict)

    if from_file is not None:
        print("* Label corpus in place")
        filename = keywords_path[:-4] + "_labeled{:s}.tsv".format(postfix)
        encoded = encoder.transform(mentions)
        contents = ["\t".join([str(itr_l), itr_c[1], itr_c[2]]) for itr_l, itr_c in zip(encoded, contents)]
        print(contents[732315])
        write_to_file(filename, contents)
        # contents['label'] = encoded
        # contents.to_csv(filename, sep="\t", header=False, index=False)
        # print(contents['mention'][732315])
        print("Converted to file: {:s}".format(filename))

def replace_labels(keywords_path, corpus, labels, output, subwords=None, mode="MULTI",
                   duplicate=True, thread=5, limit=None, tag=None):
    """

    Arguments:
        keywords_path():
        output():
        corpus():
        thread():
    """
    postfix = ("_" + tag) if tag is not None else ""
    if output is None:
        output = corpus[:-4] + "_labeled{0}{1}.tsv"\
        .format("_subwords" if subwords is not None else "", postfix)

    print("Replacing mentions with their labels:")
    print(" - Mention: {:s}".format(mode))
    print(" - Duplicate: {0}".format(duplicate))
    print()
    # Load lines from corpus
    raw_data = readlines(corpus, limit=limit)

    print()
    # Load keywords and labels
    # Used for matching each mention and its corresponding types (text)
    mentions = merge_dict(keywords_path)
    # Used for matching each type to its corresponding labels (int)
    print("Loading labels dictionary from file: {:s}".format(labels))
    contents = json.load(open(labels, "r"))
    print(" - Total number of labels: {0}".format(len(contents)))
    print()

    if subwords is not None:
        # Used for matching each type to its corresponding labels (int)
        print("Loading labels dictionary from file: {:s}".format(labels))
        subword_dict = json.load(open(subwords, "r"))
        print(" - Total number of labels: {0}".format(len(contents)))
        print()
    else:
        subword_dict = None

    # Threading
    param = (mentions, contents, subword_dict, mode, duplicate)
    result = generic_threading(thread, raw_data, keywords_as_labels, param)

    # Write result to file
    write_to_file(output, result)

def acquire_statistic(corpus, keywords_path, output=None, tag=None):
    """
    """
    postfix = ("_" + tag) if tag is not None else ""
    if output is None:
        output = corpus[:-4] + "_stat{:s}.json".format(postfix)

    # Load lines from corpus
    raw_data = readlines(corpus, limit=None)
    # [sentence] \t [mentions]
    raw_data = [[itr[:itr.find("\t")], itr[itr.find("\t") + 1:]]
                for itr in raw_data]
    df = pd.DataFrame(raw_data, columns=["CORPUS", "MENTIONS"])

    mentions = [itr.split("\t") for itr in df["MENTIONS"].as_matrix()]
    mentions = list(chain.from_iterable(mentions))
    del df

    # Count occurences
    print("Counting the occurences of labels in the dataset.")
    stat = Counter(mentions)

    # Save statistics to file
    write_to_file(output, stat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("keywords_path", type=str,
                        help="Path to where mention dictionaries are saved.")
    parser.add_argument("--model", type=str, help="Output encoder name.")
    parser.add_argument("--output", type=str, help="Sentences with key words")
    parser.add_argument("--tag", type=str, help="Make tags on the files.")
    parser.add_argument("--from_file", action="store_true", help="Load just from single file.")
    parser.add_argument("--mode", choices=["SINGLE", "MULTI"], \
                        nargs='?' , default="MULTI", help="Single mention or \
                        multi-mentions per sentence.")
    #
    parser.add_argument("--stat", action="store_true",
                        help="Acquire statistic about the amount of data in mentions.")
    #
    parser.add_argument("--find_parents", action="store_true",
                        help="Find parents for all types.")
    #
    parser.add_argument("--replace", action="store_true", help="Replace labels.")
    parser.add_argument("--labels", help="Points to data/label.json \
                        (when replacing labels).")
    parser.add_argument("--trim", action="store_true", help="Use trimmed hierarchy tree.")
    parser.add_argument("--no_duplicate", action="store_false",
                        help="Do not duplicate sentences if multiple mentions are found.")
    parser.add_argument("--corpus", help="Input labeled sentences to be replaced.")
    parser.add_argument("--subwords", help="Subword information to be added.")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    parser.add_argument("--limit", type=int, help="Number of maximum lines to load.")

    args = parser.parse_args()

    if args.replace:
        replace_labels(args.keywords_path, args.corpus, args.labels, args.output,
                       args.subwords, args.mode, args.no_duplicate, args.thread,
                       args.from_file, args.limit, args.tag)
    elif args.stat:
        acquire_statistic(args.corpus, args.keywords_path, args.output)
    else:
        fit_encoder(args.keywords_path, args.model, args.trim, args.from_file, args.output, args.tag)