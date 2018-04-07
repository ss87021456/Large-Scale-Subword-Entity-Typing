import os
import argparse
import json
from pprint import pprint
from tqdm import tqdm
from itertools import chain
from utils import write_to_file, keyword_in_sentences, readlines, generic_threading


# python src/recognize_sentences.py data/smaller_preprocessed_sentence.txt data/ --thread=10

def recognize_sentences(corpus, keywords_path, mode, split, validation, testing,
                        trim=True, label=False, output=None, thread=None):
    """
    Arguments:
        corpus(str): Path to the corpus file.
        keywords_path(str): Path to where keywords dictionaries are.
        thread(int): Number of thread to process.
        output(str): Path to the output file.
    """
    # output name
    if output is None:
        output = corpus[:-4] + "_keywords.tsv"

    # Fetch all dictionaries names
    # *** TO BE REVISED ***
    if not keywords_path.endswith("/"):
        keywords_path += "/"
    # 
    label_type = "_trimmed.json" if trim else "_leaf.json"
    files = [itr for itr in os.listdir(keywords_path) if itr.endswith(label_type)]
    # Load and merge all dictionaries
    print("Loading keywords from {:d} dictionaries".format(len(files)))
    for itr in files:
        print("- {:s}".format(itr))
    entity = dict()
    for itr in files:
        entity.update(json.load(open(keywords_path + itr, "r")))

    # Merge the keywords
    keywords_file = "data/keywords.json"
    write_to_file(keywords_file, entity)

    # Load lines from corpus
    raw_data = readlines(corpus, limit=None)

    # Threading
    keywords = list(entity.keys())
    # param = (keywords, mode, label)
    param = (keywords, mode)
    result = generic_threading(thread, raw_data, keyword_in_sentences, param)

    # write all result to file
    if split:
        amount = sum([len(itr) for itr in result])
        train_amt = amount * (1 - validation - testing)
        valid_amt = amount * validation + train_amt
        test_amt = amount * testing + valid_amt
        ### SHUFFLE ###
        # Add label
        write_to_file(output[:-4] + "_train.tsv", result[:train_amt])
        write_to_file(output[:-4] + "_validation.tsv", result[train_amt:valid_amt])
        write_to_file(output[:-4] + "_test.tsv", result[valid_amt:test_amt])
    else:
        write_to_file(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("keywords_path", help="Put all dictionaries \
                         end with \"_leaf.json\".")
    # optional arguments
    parser.add_argument("--mode", choices=["SINGLE", "MULTI"], \
                        nargs='?' , const="SINGLE", help="Single mention or \
                        multi-mentions per sentence.")
    parser.add_argument("--split", action="store_true", help="Split the dataset.")
    parser.add_argument("--validation", nargs='?', const=0.1, type=float,
                        help="The ratio of validation dataset when --split is given.")
    parser.add_argument("--testing", nargs='?', const=0.1, type=float,
                        help="The ratio of testing dataset when --split is given.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores")
    parser.add_argument("--label", action="store_true",
                        help="Replace entity name with labels.")
    parser.add_argument("--trim", action="store_true", 
                        help="Use trimmed hierarchy tree labels.")

    args = parser.parse_args()

    recognize_sentences(args.corpus, args.keywords_path, args.mode, args.split,
                        args.validation, args.testing, args.trim, args.label,
                        args.output, args.thread)
