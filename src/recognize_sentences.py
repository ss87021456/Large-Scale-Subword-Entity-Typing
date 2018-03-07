import os
import argparse
import json
from pprint import pprint
from tqdm import tqdm
from itertools import chain
from utils import keyword_in_sentences, readlines, string_file_io, generic_threading


# python src/recognize_sentences.py data/smaller_preprocessed_sentence.txt data/ --thread=10

def recognize_sentences(corpus, keywords_path, output=None, thread=None):
    """
    Arguments:
        corpus(str): Path to the corpus file.
        keywords_path(str): Path to where keywords dictionaries are.
        thread(int): Number of thread to process.
        output(str): Path to the output file.
    """
    # output name
    if output is None:
        output = corpus[:-4] + "_keywords.tsv"

    # Fetch all dictionaries names
    # *** TO BE REVISED ***
    if not keywords_path.endswith("/"):
        keywords_path += "/"
    files = [itr for itr in os.listdir(keywords_path) if itr.endswith("_leaf.json")]
    # Open and merge all dictionaries
    print("Loading keywords from {:d} dictionaries".format(len(files)))
    entity = dict()
    for itr in files:
        entity.update(json.load(open(keywords_path + itr, "r")))

    # Merge the keywords
    keywords_file = "data/keywords.json"
    print("Saving keywords to file...")
    with open(keywords_file, 'w') as fp:
        json.dump(entity, fp, sort_keys=True, indent=4)
    print("File saved in {:s}".format(keywords_file))

    # Load lines from corpus
    raw_data = readlines(corpus, limit=150)

    # Threading
    keywords = list(entity.keys())
    param = (keywords,"MULTI")
    result = generic_threading(thread, raw_data, keyword_in_sentences, param)

    # write all result to file
    string_file_io(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("keywords_path", help="Put all dictionaries \
                         end with \"_leaf.json\".")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 

    args = parser.parse_args()

    recognize_sentences(args.corpus, args.keywords_path, args.output, args.thread)
