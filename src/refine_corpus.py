import re
import argparse
from tqdm import tqdm
from pprint import pprint
from itertools import chain
from utils import generic_threading


# Define shared objects
# refine_dict = dict()
refine_list = list()
keys = None

def threading_refine(thread_idx, data):
    """
    """
    global refine_list
    # global keys
    #
    result = list()
    for article in tqdm(data, position=thread_idx):
        # refine the corpus
        # Find contents within parentheses 
        contents = re.findall(r'\(.*?\)', article)
        for itr_content in contents:
            for itr_tag in refine_list:
                found = re.findall(itr_tag[0], itr_content)
                if len(found) != 0:
                    article = article.replace(itr_content, itr_tag[1])
                else:
                    pass
        #
        #
        result.append(article)

    return result

def refine_corpus(args):
    """
    """
    # load replacement list
    global refine_list
    with open(args.dict_file) as f_list:
        lines = f_list.read().splitlines()
        for itr in lines:
            refine_list.append(itr.split('\t'))
    #
    with open(args.corpus, "r") as f_cor, open(args.output, "w") as f_out:
        # Acquire all sentences
        raw_data = f_cor.read().splitlines()[:20]
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
    parser.add_argument("dict_file", help="Containing replacement details\
                         with file extension \".json\".")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # check dict file name
    assert args.dict_file.endswith(".tsv")
    refine_corpus(args)