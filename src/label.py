import argparse
import json
import numpy as np
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from utils import write_to_file, readlines, generic_threading, keywords_as_labels, merge_dict


# python src/label.py data/ --as_dict
# python src/label.py data/ --labels=data/label.json --replace --corpus=data/smaller_preprocessed_sentence_keywords.tsv --thread=10

def fit_encoder(keywords, model=None, output=None, as_dict=False):
    """
    Arguments:
        keywords(str): Path to directory of keywords dictionary.
        one_hot(bool): Use one-hot encoding.
        model(str): Label Encoder save filename.
        output(str): Path to the output file.
    """
    file_tsv = "model/label_list.tsv"
    if model is None:
        model = "model/label_encoder.npy"
    if output is None:
        output = "data/label.json"

    # Load keywords
    print("Loading keywords from file: {:s}".format(keywords))
    contents = merge_dict(keywords)
    # contents = json.load(open(keywords, "r"))
    print("{0} keywords are loaded".format(len(contents)))

    # Unique the mentions
    mentions = list(contents.values())
    unique_types = list(np.unique(list(chain.from_iterable(mentions))))
    print("Total number of unique types: {0}".format(len(unique_types)))

    # Label Encoder
    encoder = LabelEncoder()
    print("Fitting LabelEncoder with unique types...")
    encoder.fit(unique_types)
    print("Save LabelEncoder to file {0}".format(model))
    np.save(model, encoder.classes_)

    # Encode the types
    # Consider use multi-hot encoder?
    print("Encoding unique types...")
    codes = encoder.transform(unique_types)

    # Save the labels and names as json
    print("Saving encoded result to file...")
    with open(file_tsv, "w") as f:
        for entity_type, code in zip(unique_types, codes):
            f.write(str(code) + "\t" + entity_type + "\n")
    print("Encoded result saved to {0}".format(file_tsv))

    if as_dict:
        output_dict = dict(zip(unique_types, [int(itr) for itr in codes]))
        write_to_file(output, output_dict)

def replace_labels(keywords, labels, output, replace, corpus, thread):
    """

    Arguments:
        keywords():
        output():
        replace():
        corpus():
        thread():
    """
    if output is None:
        output = corpus[:-4] + "_labeled.tsv"
    # Load keywords and labels
    print("Loading mentions dictionary from file: {:s}".format(keywords))
    #mentions = json.load(open(keywords, "r"))
    mentions = merge_dict(keywords)
    print("{0} mentions are loaded".format(len(mentions)))

    print("Loading labels dictionary from file: {:s}".format(labels))
    contents = json.load(open(labels, "r"))
    print("{0} labels are loaded".format(len(contents)))

    # Load lines from corpus
    raw_data = readlines(corpus, limit=None)

    # Threading
    param = (mentions, contents, "SINGLE")
    result = generic_threading(thread, raw_data, keywords_as_labels, param)

    # Write result to file
    write_to_file(output, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("keywords", help="Points to data/")
    parser.add_argument("--model", help="Output encoder name.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("-d", "--as_dict", action="store_true",
                        help="Save label table as dictionary.")
    #
    parser.add_argument("--labels", help="Points to data/label.json \
                        (when replacing labels).")
    parser.add_argument("--replace", action="store_true", help="Replace labels.")
    parser.add_argument("--corpus", help="Input labeled sentences to be replaced.")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 

    args = parser.parse_args()

    if args.replace:
        replace_labels(args.keywords, args.labels, args.output, args.replace, args.corpus, args.thread)
    else:
        fit_encoder(args.keywords, args.model, args.output, args.as_dict)