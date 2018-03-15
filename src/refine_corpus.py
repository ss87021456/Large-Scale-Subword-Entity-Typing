import argparse
from tqdm import tqdm
from itertools import chain
from utils import write_to_file, load_rules, readlines, generic_threading, corpus_cleanup


# python src/refine_corpus.py data/smaller_preprocessed_cleaned.txt src/refine_rules/ --thread=5

def refine_corpus(corpus, rule_path, output=None, thread=None):
    """
    Clean up the given corpus according to the rules defined in the files.
    This method utilizes multithreading to accelerate the process.

    Arguments:
        corpus(str): Path to the corpus file.
        rule_path(str): Path to where "parentheses.tsv" and 
            "refine_list.tsv" are.
        thread(int): Number of thread to process.
        output(str): Path to the output file.
    """
    if output is None:
        output = corpus[:-4] + "_cleaned.txt"
    if not rule_path.endswith("/"):
        rule_path += "/"

    # Load rule files
    file_p = rule_path + "parentheses.tsv"
    file_r = rule_path + "refine_list.tsv"
    parentheses = load_rules(file_p)
    refine_list = load_rules(file_r)

    # Acquire the corpus (skip first line)
    raw_data = readlines(corpus)

    # Threading
    param = (parentheses, refine_list)
    result = generic_threading(thread, raw_data, corpus_cleanup, param)
    
    # Write all result to file
    write_to_file(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("rule_path", help="Path to the rules defined in \
                        \"parentheses.tsv\" and \"refine_list.tsv\".")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run. [Default: 2 * number_of_cores]")

    args = parser.parse_args()

    refine_corpus(args.corpus, args.rule_path, args.output, args.thread)
