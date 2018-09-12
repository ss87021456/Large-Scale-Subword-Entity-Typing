import argparse
import json
import numpy as np
from itertools import chain
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from utils import write_to_file, readlines, generic_threading
from utils import keywords_as_labels, merge_dict, mark_positions
import pandas as pd
from collections import Counter
import csv
"""
For KBP partial dataset
python src/label.py ../share/kbp_ascii.tsv --from_file --tag=kbp --fit
python src/label.py ../share/kbp_ascii.tsv --labels=data/label_kbp.json \
--desc=wordnet_desc.json --from_file --replace

For PubMed smaller.tsv
python src/label.py data/ --trim --fit
python src/label.py data/ --labels=data/label.json --replace \
--corpus=data/smaller_preprocessed_sentence.tsv --subwords=data/subwords.json --thread=10

Check statistical information of the data
python src/label.py data/ --corpus=data/smaller_preprocessed_sentence.tsv --stat
"""


def fit_encoder(keywords_path,
                model=None,
                trim=True,
                from_file=False,
                thread=10,
                output=None,
                tag=None):
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
        labels = [itr[0] for itr in contents]
    else:
        contents = merge_dict(keywords_path, trim=trim)
        labels = list(contents.values())
        labels = list(chain.from_iterable(labels))

    # LabelEncoder
    print("Initializing LabelEncoder for encoding unique types...")
    encoder = LabelEncoder()

    unique_types = list(np.unique(labels))
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


def replace_labels(keywords_path,
                   corpus,
                   labels,
                   output,
                   subwords=None,
                   mode="MULTI",
                   from_file=False,
                   desc=None,
                   duplicate=True,
                   limit=None,
                   thread=5,
                   tag=None):
    """
    Replace text label to integer label using the preprocessed dicts/tokenizers
    from method "fit_encoder"
    Arguments:
        keywords_path(str):
        corpus(str): 
        labels(str): Path to the label dictionary (text to int)
        output(str): Output filename
        subwords(str): Path to the subword dictionary.
        mode(str): Indicating the labeling support single/multiple label per
            instance, choice ["SINGLE", "MULTI"].
        from_file(bool): Add label to the file itself.
        desc(str): Path to description file.
        duplicate(bool): Duplicate context if multiple labels exist in one
            instance if asserted.
        limit(int): Maximum number of lines to load from input file.
        thread(int): Number of thread to process.
        tag(str): Add additional tag to all input/output file.
    """
    postfix = ("_" + tag) if tag is not None else ""
    if output is None and not from_file:
        output = corpus[:-4] + "_labeled{0}{1}.tsv"\
        .format("_subwords" if subwords is not None else "", postfix)
    else:
        output = keywords_path[:-4] + "_labeled{:s}.tsv".format(postfix)

    if desc is not None:
        descriptions = json.load(open(desc, "r"))
    # Load lines from corpus
    print("Adding labels to the dataset according to their mentions:")
    print(" - Mention: {:s}".format(mode))
    print(" - Duplicate: {0}\n".format(duplicate))

    # Used for matching each type to its corresponding labels (int)
    print("Loading labels dictionary from file: {:s}".format(labels))
    lookup = json.load(open(labels, "r"))
    print(" - Total number of labels: {0}\n".format(len(lookup)))

    if from_file:
        print("* Label corpus in place")
        contents = readlines(keywords_path, limit=limit, delimitor="\t")
        # Mark position of the mention in the contexts
        print(" * Marking mention in each instance for indicators.")
        contents = mark_positions(thread_idx=0, data=contents)
        # Add label descriptions if given
        if desc is not None:
            print(" * Adding label descriptions to the dataset")
            descriptions = json.load(open(desc, "r"))
            contents = [(itr + [descriptions[itr[0]]["definition"]])
                        for itr in contents]

        # Replace types (text) to labels (int)
        encoded = [lookup[itr[0]] for itr in contents]
        # [NOTE] Single-threading seems enough for now (2M entries in 20 seconds)
        # contents = generic_threading(thread, contents, mark_positions)
        contents = [
            "\t".join([str(itr_l)] + itr_c[1:])
            for itr_l, itr_c in zip(encoded, contents)
        ]

        write_to_file(output, contents)
        # contents["label"] = encoded
        # contents.to_csv(output, sep="\t", header=False, index=False)
        print("Converted to file: {:s}".format(output))

    else:
        raw_data = readlines(corpus, limit=limit)
        # Load keywords and labels
        # Used for matching each mention and its corresponding types (text)
        mentions = merge_dict(keywords_path)

        if subwords is not None:
            # Used for matching each type to its corresponding labels (int)
            print("Loading labels dictionary from file: {:s}".format(labels))
            subword_dict = json.load(open(subwords, "r"))
            print(" - Total number of labels: {0}".format(len(lookup)))
            print()
        else:
            subword_dict = None

        # Threading
        param = (mentions, lookup, subword_dict, mode, duplicate)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keywords_path",
        type=str,
        help="Path to where mention dictionaries are saved.")
    parser.add_argument("--model", type=str, help="Output encoder name.")
    parser.add_argument("--output", type=str, help="Sentences with key words")
    parser.add_argument("--desc", type=str, help="Label descriptions.")
    parser.add_argument("--tag", type=str, help="Make tags on the files.")
    parser.add_argument(
        "--from_file", action="store_true", help="Load just from single file.")
    parser.add_argument(
        "--mode",
        choices=["SINGLE", "MULTI"],
        nargs="?",
        default="MULTI",
        help="Single mention or multi-mentions per sentence.")
    #
    parser.add_argument(
        "--stat",
        action="store_true",
        help="Acquire statistic about the amount of data in mentions.")
    #
    parser.add_argument(
        "--find_parents",
        action="store_true",
        help="Find parents for all types.")
    #
    parser.add_argument(
        "--replace", action="store_true", help="Replace labels.")
    parser.add_argument(
        "--fit", action="store_true", help="Fit labels with encoder.")
    parser.add_argument(
        "--labels", help="Points to data/label.json (when replacing labels).")
    parser.add_argument(
        "--add_desc", action="store_true", help="Add label descriptions.")
    parser.add_argument(
        "--trim", action="store_true", help="Use trimmed hierarchy tree.")
    parser.add_argument(
        "--no_duplicate",
        action="store_false",
        help="Do not duplicate sentences if multiple mentions are found.")
    parser.add_argument(
        "--corpus", help="Input labeled sentences to be replaced.")
    parser.add_argument("--subwords", help="Subword information to be added.")
    parser.add_argument(
        "--thread",
        type=int,
        help="Number of threads to run, default: 2 * number_of_cores")
    parser.add_argument(
        "--limit", type=int, help="Number of maximum lines to load.")

    args = parser.parse_args()

    if args.replace:
        replace_labels(args.keywords_path, args.corpus, args.labels,
                       args.output, args.subwords, args.mode, args.from_file,
                       args.desc, args.no_duplicate, args.limit, args.thread,
                       args.tag)
    elif args.stat:
        acquire_statistic(args.corpus, args.keywords_path, args.output)
    elif args.fit:
        fit_encoder(args.keywords_path, args.model, args.trim, args.from_file,
                    args.thread, args.output, args.tag)
    else:
        print("No job to be done.")
