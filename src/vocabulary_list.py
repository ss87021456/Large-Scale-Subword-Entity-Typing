import argparse
from string import punctuation
from collections import Counter
from itertools import chain
from pprint import pprint
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool


def threading_split(thread_idx, data):
    linewords = list()
    for itr in tqdm(data, position=thread_idx):
        linewords.append(itr.translate(punctuation).lower().split())
    return list(chain.from_iterable(linewords))

def vocabulary(args):
    with open(args.file) as f:
        raw_data = f.read().splitlines()
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
        print(" - Begin threading...")
        # Threading
        with Pool(processes=n_threads) as p:
            result = p.starmap(threading_split, thread_data)
            # result = p.apply_async(threading_split, thread_data).get()
        # count occurance
        print("\n" * n_threads)
        print("Counting occurance...")
        voc_list = Counter(chain.from_iterable(result))
        # pprint(voc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input sentences to be recognized.")
    parser.add_argument("-t", "--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 
    """
    parser.add_argument("found", help="Sentences with key words")
    parser.add_argument("dict_path", help="Put all dictionaries \
                         with extension \".json\".")
    
    """
    args = parser.parse_args()

    vocabulary(args)
