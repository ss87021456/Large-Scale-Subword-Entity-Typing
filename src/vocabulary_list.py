import argparse
import json
from string import punctuation
from collections import Counter
from itertools import chain
from tqdm import tqdm
from utils import vprint, vpprint, generic_threading


def threading_split(thread_idx, data):
    """
    """
    linewords = list()
    desc = "Thread {:2d}".format(thread_idx + 1)
    for article in tqdm(data, position=thread_idx, desc=desc):
    # for article in data:
        # replace tabs as spaces
        article = article.replace("\t", " ")
        # skip PMID
        vocabulary = article.translate(punctuation).lower().split()[1:]
        linewords.append(vocabulary)

    return list(chain.from_iterable(linewords))

def vocabulary(args):
    """
    """
    with open(args.file) as f:
        print("Loading corpus from file {:s}".format(args.file))
        raw_data = f.read().splitlines()[1:]
    # Threading
    result = generic_threading(args.thread, raw_data, threading_split)
    # count occurance
    print("Counting occurance...")
    voc_list = Counter(chain.from_iterable(result))

    # Save vocabulary to file
    print("Saving vocabulary list to file...")
    with open(args.vocb, 'w') as fp:
        json.dump(voc_list, fp, sort_keys=True, indent=4)
    print("File saved in {:s}".format(args.vocb))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input sentences to be recognized.")
    parser.add_argument("vocb", help="File name for vocabulary list to be saved.")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    """
    parser.add_argument("found", help="Sentences with key words")
    parser.add_argument("dict_path", help="Put all dictionaries \
                         with extension \".json\".")
    """
    args = parser.parse_args()

    vocabulary(args)
