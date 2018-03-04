import multiprocessing
from multiprocessing import Pool, cpu_count
from pprint import pprint
from tqdm import tqdm
from string import punctuation
from itertools import chain
import re


def vprint(msg, verbose=False):
    """
    Verbose print implementation.

    Arguments:
        msg(str): Message to 
        verbose(bool): 
    """
    if verbose:
        print(msg)
    else:
        pass

def vpprint(msg, verbose=False):
    """
    """
    if verbose:
        pprint(msg)
    else:
        pass

def load_rules(file):
    """
    .tsv files ONLY
    """
    assert file.endswith(".tsv")
    rules = list()
    with open(file, "r") as f:
        lines = f.read().splitlines()
        for itr in lines:
            rules.append(itr.split("\t"))
    #
    return rules

def split_data(data, n_slice):
    """
    """
    # Slice data for each thread
    print(" - Slicing data for threading...")
    per_slice = int(len(data) / n_slice)
    partitioned_data = list()
    for itr in range(n_slice):
        # Generate indices for each slice
        idx_begin = itr * per_slice
        # last slice may be larger or smaller
        idx_end = (itr + 1) * per_slice if itr != n_slice - 1 else None
        #
        partitioned_data.append((itr, data[idx_begin:idx_end]))
    #
    return partitioned_data

def generic_threading(n_jobs, data, method, shared_obj=None, shared=False):
    """
    """
    # Threading settings
    n_cores = cpu_count()
    n_threads = n_cores * 2 if n_jobs == None else n_jobs
    print("Number of CPU cores: {:d}".format(n_cores))
    print("Number of Threading: {:d}".format(n_threads))
    #
    thread_data = split_data(data, n_threads)
    #
    print(" - Begin threading...")
    # Threading
    with Pool(processes=n_threads) as p:
        if not shared:
            result = p.starmap(method, thread_data)
        else:
            p.starmap(method, thread_data)
    #
    print("\n" * n_threads)
    print("All threads completed.")
    return result if not shared else None


def punctuation_cleanup(thread_idx, data, rules, mode):
    """
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    ########### EXCEPTION HANDLING ###########
    # assert mode 

    # global rules
    linewords = list()
    for article in tqdm(data, position=thread_idx, desc=desc):
    # for article in data:
        # replace tabs as spaces
        article = article.replace("\t", " ")
        # SPLIT_WORDS: used in finding vocabularies
        if mode == 'SPLIT_WORDS':
            # skip PMID
            vocabulary = article.translate(punctuation).lower().split()
            # cleanup some redundant punctuation
            for itr in rules:
                pattern, _ = itr
                # symbols at the end
                if pattern.startswith("*"):
                    symbol = pattern[1:]
                    vocabulary = [i[:-len(symbol)] if i.endswith(symbol)
                                  and not "-" in i else i
                                  for i in vocabulary]
                # symbols in the beginning
                elif pattern.endswith("*"):
                    symbol = pattern[:-1]
                    vocabulary = [i[len(symbol):] if i.startswith(symbol)
                                  and not "-" in i else i
                                  for i in vocabulary]
                else:
                    vocabulary = [i.replace(pattern, "") for i in vocabulary]

            linewords.append(vocabulary)
        # PRELIMINARY:
        elif mode == 'PRELIMINARY':
            for itr in rules:
                pattern, replacement = itr
                replacement = replacement[1:] if "*" in replacement
                found = re.findall(pattern, article)
                for itr_found in found:
                    article.replace(itr_found)
        else:
            print("Invalid mode type: {0}".format(mode))


    # result[thread_idx] = list(chain.from_iterable(linewords))
    return list(chain.from_iterable(linewords))