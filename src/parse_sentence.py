import argparse
import nltk.data
from tqdm import tqdm
from utils import readlines, write_to_file, generic_threading

# python parse_sentence.py data/smaller_preprocessed.tsv --thread=10

def parallel_parse(thread_idx, data, tokenizer):
    """

    Arguments:
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
        tab_index = content[0].find('\t')
        content[0] = content[0][tab_index+1:] # skip PMID\t
        for element in content:
            result.append(element)

    return result

def parse_sentences(corpus, output=None, thread=None, limit=None):
    """
    """
    if output is None:
        output = corpus[:-4] + "_sentence.txt"

    # Load corpus
    raw_data = readlines(corpus, begin=1, limit=limit) # skip first line

    # Load tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Threading
    param = (tokenizer, )
    result = generic_threading(thread, raw_data, parallel_parse, param)

    # Write all result to file
    write_to_file(output, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="File to be parsed.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores")
    parser.add_argument("--limit", type=int, help="Number of maximum lines to load.")

    args = parser.parse_args()

    parse_sentences(args.corpus, args.output, args.thread, args.limit)
