import argparse
import json
from collections import Counter
from itertools import chain
from tqdm import tqdm
from utils import load_rules, generic_threading, punctuation_cleanup


def init_share_mem(n_threads):
    global result
    result = [None for _ in range(args.thread)]
    print(id(result))

def vocabulary(args):
    """
    """
    # global rules
    rules = load_rules(args.rule)
    
    with open(args.file) as f:
        print("Loading corpus from file {:s}".format(args.file))
        raw_data = f.read().splitlines()[:20]
    # Threading
    # init_share_mem(args.thread)
    # generic_threading(args.thread, raw_data, punctuation_cleanup, shared=True)
    result = punctuation_cleanup(0, raw_data, rules, mode='SPLIT_WORDS')
    # count occurance
    print("Counting occurance...")
    # voc_list = Counter(chain.from_iterable(result))
    voc_list = Counter(result)

    # Save vocabulary to file
    print("Saving vocabulary list to file...")
    with open(args.vocb, 'w') as fp:
        json.dump(voc_list, fp, sort_keys=True, indent=4)
    print("File saved in {:s}".format(args.vocb))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input sentences to be recognized.")
    parser.add_argument("vocb", help="File name for vocabulary list to be saved.")
    parser.add_argument("rule", help="Rules for vocabulary list to be cleaned up.")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    """
    parser.add_argument("found", help="Sentences with key words")
    parser.add_argument("dict_path", help="Put all dictionaries \
                         with extension \".json\".")
    """
    args = parser.parse_args()

    vocabulary(args)
