import os
import argparse
import json
import nltk.data
from pprint import pprint
from tqdm import tqdm
from itertools import chain
from utils import write_to_file, keyword_in_sentences, readlines, generic_threading, merge_dict
"""
python src/recognize_sentences.py data/smaller_preprocessed.tsv data/ --trim --mode=MULTI --thread=10
"""


def parallel_split(thread_idx, data, tokenizer):
    """
    Args:
        thread_idx()
        data()
        tokenizer():

    Return:
        result():
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    for article in tqdm(data, position=thread_idx, desc=desc):
        content = tokenizer.tokenize(article)
        tab_index = content[0].find("\t")
        content[0] = content[0][tab_index + 1:]  # skip PMID\t
        for element in content:
            result.append(element)

    return result


def recognize_sentences(corpus,
                        keywords_path,
                        mode,
                        trim=True,
                        label=False,
                        output=None,
                        thread=None,
                        limit=None):
    """
    Arguments:
        corpus(str): Path to the corpus file.
        keywords_path(str): Path to where keywords dictionaries are.
        thread(int): Number of thread to process.
        output(str): Path to the output file.
    """
    # output name
    if output is None:
        output = corpus[:-4] + "_sentences.tsv"

    # Decompose corpus to sentences and each as one datum
    # Load corpus (skip first line for PubMed smaller version)
    raw_data = readlines(corpus, begin=1, limit=limit)
    # Load tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # Threading
    param = (tokenizer, )
    context = generic_threading(thread // 2, raw_data, parallel_split, param)
    context = list(chain.from_iterable(context))
    del raw_data

    print()
    print("Recognize mentions in sentences (mode: {:s})".format(mode))

    # Load all mentions
    entity = merge_dict(keywords_path, trim=trim)

    # Threading
    keywords = list(entity.keys())
    param = (keywords, mode)
    result = generic_threading(thread, context, keyword_in_sentences, param)

    # write all result to file
    write_to_file(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument(
        "keywords_path", help="Put all dictionaries end with \"_leaf.json\".")
    # optional arguments
    parser.add_argument(
        "--mode",
        choices=["SINGLE", "MULTI"],
        nargs='?',
        default="MULTI",
        help="Single mention or multi-mentions per sentence.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument(
        "--thread",
        type=int,
        help="Number of threads to run, default: 2 * number_of_cores")
    parser.add_argument(
        "--label",
        action="store_true",
        help="Replace entity name with labels.")
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Use trimmed hierarchy tree labels.")
    parser.add_argument(
        "--limit", type=int, help="Number of maximum lines to load.")
    args = parser.parse_args()

    recognize_sentences(args.corpus, args.keywords_path, args.mode, args.trim,
                        args.label, args.output, args.thread, args.limit)
