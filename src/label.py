import argparse
import json
import numpy as np
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from utils import string_file_io


# python src/label.py FIT_ENCODER data/keywords.json
# python src/label.py mode=ENCODE data/keywords.json

def fit_encoder(keywords, one_hot=False, model=None, output=None):
    """
    Arguments:
        keywords(str): Path to keywords dictionary.
        one_hot(bool): Use one-hot encoding.
        model(str): Label Encoder save filename.
        output(str): Path to the output file.
    """
    if model is None:
        model = "model/label_encoder.npy"
    if output is None:
        output = "model/label.tsv"

    # Load keywords
    print("Loading keywords from file: {:s}".format(keywords))
    contents = json.load(open(keywords, "r"))
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
    print("Saving encoded result to file...")
    with open(output, "w") as f:
        for entity_type, code in zip(unique_types, codes):
            if one_hot:
                f.write(str(code) + "\t" + entity_type + "\n")
            else:
                f.write(str(code) + "\t" + entity_type + "\n")
    print("Encoded result saved to {0}".format(output))

def encode_entity_types():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Indicating the model of encoder.\
                        FIT_ENCODER or ENCODE")
    parser.add_argument("keywords", help="Points to data/keywords.json.")
    parser.add_argument("--model", help="Output encoder name.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("-oh", "--one_hot", help="Use multi-hot labeling.")

    args = parser.parse_args()

    if args.mode == 'FIT_ENCODER':
        fit_encoder(args.keywords, args.one_hot, args.model, args.output)
    elif args.mode == 'ENCODE':
        encode_entity_types()
    else:
        print("Please specify correct mode: FIT_ENCODER or ENCODE")