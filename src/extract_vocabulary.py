import argparse
import json
from collections import Counter
from itertools import chain
from tqdm import tqdm
from utils import write_to_file, readlines, load_rules, generic_threading, punctuation_cleanup

# python src/extract_vocabulary.py data/smaller_preprocessed.tsv src/refine_rules/voc_cleanup.tsv --thread=5


def extract_vocabularies(corpus, rule, output=None, thread=None):
    """
    Extract vocabularies from the corpus, additional rules to achieve
    purer vocabularies can be defined in src/refine_rules/voc_cleanup.tsv

    Arguments:
        corpus(str): Path to the corpus file.
        rule(str): Path to the processing rule file.
        thread(int): Number of thread to process.
        output(str): Path to the output file.
    """
    if output is None:
        output = corpus[:-4] + "_vocabulary_list.json"

    # Load rules
    rules = load_rules(rule)

    # Acquire the corpus
    raw_data = readlines(corpus, limit=None)

    # Threading (TO-BE-IMPLEMENTED)
    # param = (rules, "SPLIT_WORDS")
    # generic_threading(thread, raw_data, punctuation_cleanup, param)
    result = punctuation_cleanup(0, raw_data, rules, mode='SPLIT_WORDS')

    # Counting occurance
    print("Counting occurance...")
    voc_list = Counter(result)

    # Save vocabulary to file
    write_to_file(output, voc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("rule", help="Rules to purify vocabularies.")
    parser.add_argument(
        "--output",
        help="File name for vocabulary list to\
                        be saved. [Default: vocabulary_list.json]")
    parser.add_argument(
        "--thread",
        type=int,
        help="Number of threads \
                        to run. [Default: 2 * number_of_cores]")

    args = parser.parse_args()

    extract_vocabularies(args.corpus, args.rule, args.output, args.thread)
