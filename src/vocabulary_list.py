import argparse
import json
from string import punctuation
from collections import Counter
from itertools import chain
from tqdm import tqdm
from utils import vprint, vpprint, load_rules, generic_threading

result = None
rules = None

def threading_split(thread_idx, data):
    """
    """
    global rules
    linewords = list()
    desc = "Thread {:2d}".format(thread_idx + 1)
    for article in tqdm(data, position=thread_idx, desc=desc):
    # for article in data:
        # replace tabs as spaces
        article = article.replace("\t", " ")
        # skip PMID
        vocabulary = article.translate(punctuation).lower().split()
        # cleanup some redundant punctuation
        for itr in rules:
            pattern, _ = itr
            # symbols at the end
            if pattern.startswith("*"):
                symbol = pattern[1:]
                vocabulary = [i[:-len(symbol)] if i.endswith(symbol)
                              and not "-" in i else i
                              for i in vocabulary]
                """
                for i in vocabulary:
                    if "ad-d)" in i and symbol == ")":
                        print(i)
                        exit()
                """
            # symbols in the beginning
            elif pattern.endswith("*"):
                symbol = pattern[:-1]
                vocabulary = [i[len(symbol):] if i.startswith(symbol)
                              and not "-" in i else i
                              for i in vocabulary]
            else:
                vocabulary = [i.replace(pattern, "") for i in vocabulary]
                """
                for i in vocabulary:
                    if "ad-d" in i:
                        print(vocabulary, pattern)
                        print(i.replace(pattern, ""))
                        exit()
                """
        linewords.append(vocabulary)

    # result[thread_idx] = list(chain.from_iterable(linewords))
    # result = list(chain.from_iterable(linewords))
    # print(result[thread_idx])
    # print("Thread {:d} done".format(thread_idx))
    return list(chain.from_iterable(linewords))

def init_share_mem(n_threads):
    global result
    result = [None for _ in range(args.thread)]
    print(id(result))

def vocabulary(args):
    """
    """
    global rules
    rules = load_rules(args.rule)
    
    with open(args.file) as f:
        print("Loading corpus from file {:s}".format(args.file))
        raw_data = f.read().splitlines()
    # Threading
    # init_share_mem(args.thread)
    # generic_threading(args.thread, raw_data, threading_split, shared=True)
    result = threading_split(0, raw_data)
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
