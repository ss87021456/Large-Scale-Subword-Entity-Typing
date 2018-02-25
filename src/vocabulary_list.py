import argparse
from string import punctuation
from collections import Counter
from itertools import chain
from pprint import pprint
from tqdm import tqdm
from utils import generic_threading


def threading_split(thread_idx, data):
    linewords = list()
    for itr in tqdm(data, position=thread_idx):
        linewords.append(itr.translate(punctuation).lower().split())
    return list(chain.from_iterable(linewords))

def vocabulary(args):
    with open(args.file) as f:
        raw_data = f.read().splitlines()
        # Threading
        result = generic_threading(args.thread, raw_data, threading_split)
        # count occurance
        print("\n" * n_threads)
        print("Counting occurance...")
        voc_list = Counter(chain.from_iterable(result))
        # pprint(voc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input sentences to be recognized.")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    """
    parser.add_argument("found", help="Sentences with key words")
    parser.add_argument("dict_path", help="Put all dictionaries \
                         with extension \".json\".")
    """
    args = parser.parse_args()

    vocabulary(args)
