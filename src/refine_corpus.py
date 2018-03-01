import re
import os
import argparse
from tqdm import tqdm
from pprint import pprint
from itertools import chain
from utils import load_rules, generic_threading


# Define shared objects
refine_list = list()
parentheses = list()

def threading_refine(thread_idx, data):
    """
    """
    global refine_list
    global parentheses
    desc = "Thread {:2d}".format(thread_idx + 1)
    #
    result = list()
    for article in tqdm(data, position=thread_idx, desc=desc):
        article = article[article.find("\t") + 1:] # skip PMID\t
        # refine the corpus
        # Find contents within parentheses 
        contents = re.findall(r"\(.*?\)", article)
        for itr_content in contents:
            for itr_tag in parentheses:
                # extract entry
                pattern, tag = itr_tag
                # find pattern
                found = re.findall(pattern, itr_content)
                if len(found) != 0:
                    # add redundant spaces to avoid words stay together
                    article = article.replace(itr_content, " " + tag + " ")
                else:
                    pass
        # Find and replace patterns in the article
        for itr_pattern in refine_list:
            pattern, tag = itr_pattern
            #found = re.findall(pattern, article)
            article = re.sub(pattern, " " + tag + " ", article)
        #
        result.append(article.lower())

    return result

def refine_corpus(args):
    """
    """
    # Load replacement list
    global refine_list
    global parentheses
    # Load rule files
    file_r = args.rule_path + "refine_list.tsv"
    file_p = args.rule_path + "parentheses.tsv"
    refine_list = load_rules(file_r)
    parentheses = load_rules(file_p)

    with open(args.corpus, "r") as f_cor:
        # Acquire the corpus (skip first line)
        raw_data = f_cor.read().splitlines()[1:]

    # Threading
    result = generic_threading(args.thread, raw_data, threading_refine)
    
    # Write all result to file
    with open(args.output, "w") as f_out:
        # *** TO BE REVISED, MAY CONSUME TOO MUCH MEMORY ***
        print("Writing result to file...")
        for line in tqdm(list(chain.from_iterable(result))):
            f_out.write(line + "\n")
        print("File saved in {:s}".format(args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("output", help="Sentences with key words")
    parser.add_argument("rule_path", help="Containing replacement details\
                         with file extension \".tsv\".")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    refine_corpus(args)
