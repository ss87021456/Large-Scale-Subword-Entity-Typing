import argparse
from utils import load_rules, generic_threading


rules = None


def preliminary_cleanup(corpus, rule, thread, verbose):
    # Load rules
    # global rules
    # rules = load_rules(rule)

    # load corpus
    with open(corpus, "r") as f:
        raw_data = f.read().splitlines()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("rule", help="Clean up rule.")
    parser.add_argument("--output", help="Output file name.\
                        [Default: ${INPUT_NAME}_preprocessed.tsv].")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run. [Default: 2 * number_of_cores]") 
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    preliminary_cleanup(args.corpus, args.rule, args.thread, args.verbose)