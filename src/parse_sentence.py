import nltk.data
import sys, pprint
from utils import *
# import numpy as np

filename = sys.argv[1]
save_name = filename[:-4] + '_sentence.txt'
thread_num = int(sys.argv[2])

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def parallel_parse(thread_idx, data):
    global tokenizer
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    for article in tqdm(data, position=thread_idx, desc=desc):
        content = tokenizer.tokenize(article)
        tab_index = content[0].find('\t')
        content[0] = content[0][tab_index+1:] # skip PMID\t
        for element in content:
            result.append(element)

    return result


with open(filename, 'r') as f, open(save_name, "w") as f_out:
    raw_data = f.read().splitlines()[1:] # skip first line
    # Threading
    result = generic_threading(thread_num, raw_data, parallel_parse)
    save_name = filename[:-4] + '_sentence.txt'
    # Write all result to file
    print("Writing result to file...")
    for line in tqdm(list(chain.from_iterable(result))):
        f_out.write(line + "\n")
    print("File saved in {:s}".format(save_name))

