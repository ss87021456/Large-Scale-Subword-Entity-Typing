import argparse
from utils import load_rules, generic_threading, punctuation_cleanup, readlines
from tqdm import tqdm
from itertools import chain


# python src/preliminary.py data/smaller.tsv src/refine_rules/preliminary.tsv --thread=5

def preliminary_cleanup(corpus, rule, thread, output=None):
    """
    Preliminary cleanup the corpus to make it easier for further
    processing methods. This method can be used to correct the
    missing spaces after punctuations any other customized rules
    can be added to the rule file, see punctuation_cleanup in utils
    for the formatting of the rules.

    Arguments:
        corpus(str): Path to the corpus file.
        rule(str): Path to the processing rule file.
        thread(int): Number of thread to process.
        output(str): Path to the output file.
    """
    # output name
    if output is None:
        output = corpus[:-4] + "_preprocessed.tsv"

    # Load rules
    rules = load_rules(rule)
    # Load data
    raw_data = readlines(corpus)

    # Threading
    param = (rules, "PRELIMINARY")
    result = generic_threading(thread, raw_data, punctuation_cleanup, param)

    # Write result to file
    string_file_io(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("rule", help="Cleanup rules (.tsv).")
    parser.add_argument("--output", help="Output file name. \
                        [Default Postfix: \"_preprocessed.tsv\"].")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run. [Default: 2 * number_of_cores]")
    args = parser.parse_args()

    preliminary_cleanup(args.corpus, args.rule, args.thread, args.output)
