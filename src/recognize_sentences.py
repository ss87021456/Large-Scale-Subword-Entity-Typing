import os
import argparse
import json
from pprint import pprint
from mmap import mmap
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from itertools import chain


# Used as shared memory among threads
keywords = None

def threading_search(thread_idx, data):
    """
    Sentence keyword search (called by threads)

    Arguments:
        thread_idx(int): Order of threads, used to align progressbar.
        data(list of str): Each elements in the list contains one sentence
                          of raw corpus.

    Returns:
        result(listof str): Each elements in the list contains one 
                     sentence with one or more keywords.
    """
    global keywords
    result = list()
    #
    for line in tqdm(data, position=thread_idx):
        # search for keywords
        for itr in keywords:
            if itr in line:
                # print(" - Found sentence with keyword: {0}".format(itr))
                result.append(line)
                break
            else:
                pass
    return result

def recognize_sentences(args):
    """
    """
    with open(args.sentences, "r") as f_in, open(args.found, "w") as f_out:
        # Fetch all dictionaries names
        # *** TO BE REVISED ***
        files = [itr for itr in os.listdir(args.dict_path) 
                 if itr.endswith('_leaf.json')]
        # Open and merge all dictionaries
        print("Loading keywords from {:d} dictionaries".format(len(file)))
        entity = dict()
        for itr in files:
            entity.update(json.load(open(args.dict_path + '/' + itr, "r")))
        # Acquire keys and share memory
        global keywords
        keywords = entity.keys()
        # Acquire all sentences
        raw_data = f_in.read().splitlines()
        # Threading settings
        n_cores = multiprocessing.cpu_count()
        n_threads = n_cores * 2 if args.thread == None else args.thread
        print("Number of CPU cores: {:d}".format(n_cores))
        print("Number of Threading: {:d}".format(n_threads))
        # Slice data for each thread
        print(" - Slicing data for threading...")
        per_slice = int(len(raw_data) / n_threads)
        thread_data = list()
        for itr in range(n_threads):
            # Generate indices for each slice
            idx_begin = itr * per_slice
            # last slice may be larger or smaller
            idx_end = (itr + 1) * per_slice if itr != n_threads - 1 else None
            #
            thread_data.append((itr, raw_data[idx_begin:idx_end]))
        #
        print(" - Begin threading...")
        # Threading
        with Pool(processes=n_threads) as p:
            result = p.starmap(threading_search, thread_data)
        print("\n\nAll threads completed.")
        # write all result to file
        # *** TO BE REVISED, MAY CONSUME TOO MUCH MEMORY ***
        print("Writing result to file...")
        for line in tqdm(list(chain.from_iterable(result))):
            f_out.write(line + "\n")
        print("File saved in {:s}".format(args.found))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences", help="Input sentences to be recognized.")
    parser.add_argument("found", help="Sentences with key words")
    parser.add_argument("dict_path", help="Put all dictionaries \
                         with extension \".json\".")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    recognize_sentences(args)
