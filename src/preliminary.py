import argparse
from utils import load_rules, generic_threading, punctuation_cleanup
from tqdm import tqdm
from itertools import chain


# python src/preliminary.py data/smaller.tsv src/refine_rules/preliminary.tsv --thread=5

def preliminary_cleanup(corpus, rule, thread, output=None, verbose=False):
    """
    Preliminary cleanup the corpus to make it easier for further
    processing methods. This method can be used to correct the
    missing spaces after punctuations any other customized rules
    can be added to the rule file, see punctuation_cleanup in utils
    for the formatting of the rules.

    Arguments:
        corpus(str): Path to the corpus file.
        output(str): Path to the output file.
        rule(str): Path to the processing rule file.
        thread(int): Number of thread to process.
        verbose(bool): (undefined)
    """
    # output name
    if output is None:
        output = corpus[:-4] + "_preprocessed.tsv"
    # Load rules
    rules = load_rules(rule)

    # load corpus
    with open(corpus, "r") as f:
        raw_data = f.read().splitlines()

    # Threading
    param = (rules, "PRELIMINARY")
    result = generic_threading(thread, raw_data, punctuation_cleanup, param)

    with open(output, "w") as f:
        for line in tqdm(list(chain.from_iterable(result))):
            f.write(line + "\n")
    print("File saved in {:s}".format(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("rule", help="Cleanup rules (.tsv).")
    parser.add_argument("--output", help="Output file name. \
                        [Default Postfix: \"_preprocessed.tsv\"].")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run. [Default: 2 * number_of_cores]") 
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    preliminary_cleanup(args.corpus, args.rule, args.thread, args.output, args.verbose)
