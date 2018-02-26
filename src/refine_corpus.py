import re
import os
import argparse
from tqdm import tqdm
from pprint import pprint
from itertools import chain
from utils import generic_threading


# Define shared objects
refine_list = list()
parentheses = list()
keys = None

def threading_refine(thread_idx, data):
    """
    """
    global refine_list
    global parentheses
    # global keys
    #
    result = list()
    for article in tqdm(data, position=thread_idx):
        # refine the corpus
        # Find contents within parentheses 
        contents = re.findall(r'\(.*?\)', article)
        for itr_content in contents:
            for itr_tag in parentheses:
                # extract entry
                pattern, tag = itr_tag
                # find pattern
                found = re.findall(pattern, itr_content)
                if len(found) != 0:
                    # add redundant space to avoid words stay together
                    article = article.replace(itr_content, " " + tag)
                else:
                    pass
        # Find and replace patterns in the article
        for itr_pattern in refine_list:
            pattern, tag = itr_pattern
            found = re.findall(pattern, article)
            if len(found) != 0:
                for itr_found in found:
                    # add redundant space to avoid words stay together
                    target = " " + itr_found + " "
                    article = article.replace(target, " " + tag)
            else:
                pass
        #
        result.append(article)

    return result

def refine_corpus(args):
    """
    """
    # load replacement list
    global refine_list
    global parentheses
    # Load rule files
    file_p = args.rule_path + "parentheses.tsv"
    file_r = args.rule_path + "refine_list.tsv"
    #
    with open(file_p, "r") as fp, open(file_r, "r") as fr:
        # parentheses
        lines = fp.read().splitlines()
        for itr in lines:
            parentheses.append(itr.split('\t'))
        # refine_list
        lines = fr.read().splitlines()
        for itr in lines:
            refine_list.append(itr.split('\t'))
    #
    with open(args.corpus, "r") as f_cor, open(args.output, "w") as f_out:
        # skip first line
        # Acquire the corpus
        raw_data = f_cor.read().splitlines()[1:]
        # Threading
        result = generic_threading(args.thread, raw_data, threading_refine)
        # write all result to file
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