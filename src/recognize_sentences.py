import os
import argparse
import json
from pprint import pprint
from tqdm import tqdm
from itertools import chain
from utils import readlines, string_file_io, generic_threading


# python src/recognize_sentences.py data/smaller_preprocessed_sentence.txt data/ --thread=10

def keyword_in_sentences(thread_idx, data, keywords):
    """
    Sentence keyword search (called by threads or used normally)

    Arguments:
        thread_idx(int): Order of threads, used to align progressbar.
        data(list of str): Each elements in the list contains one sentence
                          of raw corpus.

    Returns:
        result(list of str): Each elements in the list contains one 
                     sentence with one or more keywords.
    """
    # global keywords
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    #
    for line in tqdm(data, position=thread_idx, desc=desc):
        # search for keywords
        found = False
        found_keyword = list()
        for itr in keywords:
            # Append keywords to the list
            if itr.lower() in line.lower():
                found = True
                found_keyword.append(itr)
        #
        if found:
            result.append(line + "\t" + "\t".join(found_keyword))

    return result

def recognize_sentences(corpus, rule, thread, output=None, verbose=False):
    """
    """
    # output name
    if output is None:
        output = corpus[:-4] + "_keywords.tsv"

    # Fetch all dictionaries names
    # *** TO BE REVISED ***
    if not rule.endswith("/"):
        rule += "/"
    files = [itr for itr in os.listdir(rule) if itr.endswith("_leaf.json")]
    # Open and merge all dictionaries
    print("Loading keywords from {:d} dictionaries".format(len(files)))
    entity = dict()
    for itr in files:
        entity.update(json.load(open(rule + itr, "r")))

    # Load lines from corpus
    raw_data = readlines(corpus, limit=None)

    # Threading
    keywords = list(entity.keys())
    param = (keywords,)
    result = generic_threading(thread, raw_data, keyword_in_sentences, param)

    # write all result to file
    string_file_io(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("rule", help="Put all dictionaries \
                         with extension \".json\".")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    recognize_sentences(args.corpus, args.rule, args.thread, args.output, args.verbose)
