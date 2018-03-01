import os
import argparse
import json
from pprint import pprint
from tqdm import tqdm
from itertools import chain
from utils import generic_threading


# Used as shared memory among threads
keywords = None

def threading_search(thread_idx, data):
    """
    Sentence keyword search (called by threads)

    Arguments:
        thread_idx(int): Order of threads, used to align progressbar.
        data(list of str): Each elements in the list contains one sentence
                          of raw corpus.

    Returns:
        result(listof str): Each elements in the list contains one 
                     sentence with one or more keywords.
    """
    global keywords
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    #
    for line in tqdm(data, position=thread_idx, desc=desc):
        # search for keywords
        for itr in keywords:
            if itr in line:
                # print(" - Found sentence with keyword: {0}".format(itr))
                result.append(line)
                break
            else:
                pass
    return result

def recognize_sentences(args):
    """
    """
    with open(args.sentences, "r") as f_in, open(args.found, "w") as f_out:
        # Fetch all dictionaries names
        # *** TO BE REVISED ***
        files = [itr for itr in os.listdir(args.dict_path) 
                 if itr.endswith('_leaf.json')]
        # Open and merge all dictionaries
        print("Loading keywords from {:d} dictionaries".format(len(files)))
        entity = dict()
        for itr in files:
            entity.update(json.load(open(args.dict_path + '/' + itr, "r")))
        # Acquire keys and share memory
        global keywords
        keywords = entity.keys()
        # Acquire all sentences
        raw_data = f_in.read().splitlines()[:20]
        # Threading
        result = generic_threading(args.thread, raw_data, threading_search)
        # write all result to file
        # *** TO BE REVISED, MAY CONSUME TOO MUCH MEMORY ***
        print("Writing result to file...")
        for line in tqdm(list(chain.from_iterable(result))):
            f_out.write(line + "\n")
        print("File saved in {:s}".format(args.found))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences", help="Input sentences to be recognized.")
    parser.add_argument("found", help="Sentences with key words")
    parser.add_argument("dict_path", help="Put all dictionaries \
                         with extension \".json\".")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    recognize_sentences(args)
