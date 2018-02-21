import os
import argparse
import json
from pprint import pprint
from mmap import mmap
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool


keywords = None

def threading_search(thread_idx, data):
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

def get_num_lines(file_path):
    lines = 0
    with open(file_path, "r+") as fp:
        buf = mmap(fp.fileno(), 0)
        while buf.readline():
            lines +=1
    return lines

def recognize_sentences(args):
    with open(args.sentences, "r") as f_in, open(args.found, "w") as f_out:
        # load dictionarys
        files = [itr for itr in os.listdir(args.dict_path) 
                 if itr.endswith('_leaf.json')]
        #
        entity = dict()
        for itr in files:
            entity.update(json.load(open(args.dict_path + '/' + itr, "r")))
        #
        global keywords
        keywords = entity.keys()
        count = 0
        #
        raw_data = f_in.read().splitlines()
        n_cores = multiprocessing.cpu_count()
        n_threads = n_cores * 2 if args.thread == None else args.thread
        print("Number of cores available: {:d}".format(n_cores))
        print("Number of threads opended: {:d}".format(n_threads))
        # Slice data
        thread_data = list()
        per_slice = int(len(raw_data) / n_threads)
        for itr in range(n_threads):
            idx_begin = itr * per_slice
            idx_end = (itr + 1) * per_slice if itr != n_threads - 1 else None
            #
            thread_data.append((itr, raw_data[idx_begin:idx_end]))
        # Threading
        # pool = ThreadPool(n_threads)
        with Pool(processes=n_threads) as p:
            result = p.starmap(threading_search, thread_data)

        # write to file
        for per_thread in result:
            for line in per_thread:
                f_out.write(line + "\n")
        """
        for line in tqdm(f_in, total=total_lines):
            # search for keywords
            for itr in keywords:
                if itr in line:
                    # print(" - Found sentence with keyword: {0}".format(itr))
                    f_out.write(line)
                    break
                else:
                    pass
        """


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
