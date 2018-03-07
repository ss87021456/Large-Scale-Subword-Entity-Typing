import re
import random
import multiprocessing
from multiprocessing import Pool, cpu_count
from pprint import pprint
from tqdm import tqdm
from string import punctuation
from itertools import chain


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

def readlines(file, begin=None, limit=None, rand=False):
    """
    Read and split all content in the files line by line.

    Arguments:
        file(str): File to be read.
        begin(int): Index of the first line to be read.
        limit(int): Index of the last line to be read, or the amount of
            samples drawn from the dataset if rand is asserted.
        rand(bool): Randomly drawn samples from the data.
    Return:
        raw_data(list of strings): Lines from the files
    """
    print("Loading lines in the file...")
    with open(file, "r") as f:
        if rand:
            print(" - Random sample {:d} entries from data.".format(limit))
            data = f.read().splitlines()
            n_data = len(data)
            index = sorted(random.sample(list(range(n_data)), limit))
            data = [data[itr] for itr in index]
        else:
            data = f.read().splitlines()[begin:limit]
    print("Total {0} lines loaded.".format(len(data)))
    return data

def string_file_io(file, data):
    """
    Write strings to files.

    Arguments:
        file(str): Output filename.
        data(list of list): List of list of strings.
    """
    print("Writing result to file...")
    with open(file, "w") as f:
        if type(data[0]) == list:
            for itr in tqdm(list(chain.from_iterable(data))):
                f.write(itr + "\n")
        else:
            for itr in tqdm(data):
                f.write(itr + "\n")
    print("File saved in {:s}".format(file))

def split_data(data, n_slice):
    """
    Split data to minibatches with last batch may be larger or smaller.

    Arguments:
        data(ndarray): Array of data.
        n_slice(int): Number of slices to separate the data.

    Return:
        partitioned_data(list): List of list containing any type of data.
    """
    n_data = len(data)
    # Slice data for each thread
    print(" - Slicing data for threading...")
    print(" - Total number of data: {0}".format(n_data))
    per_slice = int(n_data / n_slice)
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

def generic_threading(n_jobs, data, method, param=None, shared=False):
    """
    Generic threading method.

    Arguments:
        n_jobs(int): number of thead to run the target method
        data(ndarray): Data that will be split and distributed to threads.
        method(method object): Threading target method
        param(tuple): Tuple of additional parameters needed for the method.
        shared: (undefined)

    Return:
        result(list of any type): List of return values from the method.
    """
    # Threading settings
    n_cores = cpu_count()
    n_threads = n_cores * 2 if n_jobs == None else n_jobs
    print("Number of CPU cores: {:d}".format(n_cores))
    print("Number of Threading: {:d}".format(n_threads))
    #
    thread_data = split_data(data, n_threads)
    if param is not None:
        thread_data = [itr + param for itr in thread_data]
    else:
        pass
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
    Generic clean up the given corpus, the cleanup process can be either
    corpus-level (preliminary cleanup) or finer cleanups, i.e. vocabulary-
    level (for gaining purer vocabularies).

    Arguments:
        thread_idx(int): Indicating the thread ID, used for the positioning
                         and information of the progressbar.
        data(list of str): Each entry is a corpus to be processed.
        rules(list of tuples):
        mode(str): Mode of the cleanup method, available modes are
                SPLIT_WORDS: Used in extracting the vocabularies.
                PRELIMINARY: Preliminary rules for cleaning up the corpus.
    Return:
        linewords(list of str): List of either corpus (PRELIMINARY) or
                words (SPLIT_WORDS)
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    ########### EXCEPTION HANDLING ########### (TO-BE-IMPELMENTED)
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
                pattern, tag = itr
                found = re.findall(pattern, article)
                if len(found) == 0:
                    continue
                # support * replacement
                if tag.startswith("*"):
                    tag = tag[1:]
                    for itr_found in found:
                        article = article.replace(itr_found, itr_found.replace(tag, tag + " "))
                else:
                    for itr_found in found:
                        article = article.replace(itr_found, tag)
            linewords.append(article)
        else:
            print("Invalid mode type: {0}".format(mode))

    if mode == "SPLIT_WORDS":
        linewords = list(chain.from_iterable(linewords))
    return linewords

def corpus_cleanup(thread_idx, data, parentheses, refine_list):
    """
    Used to clean up the corpus according to the predefined regex commands.
    This method can be used standalone or passed to threads to improve
    performance.

    Arguments:
        thread_idx(int): Indicating the thread ID, used for the positioning
                         and information of the progressbar.
        data(list of str): Each entry is a corpus to be processed.
        parentheses(list of tuples): List containing patterns in the
            parentheses that are to be replaced by tags.
        refine_list(list of tuples):List containig patterns to be replaced
            by tags.
    Returns:
        result(list of str): Processed version of the input "data"
    """
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
            article = re.sub(pattern, " " + tag + " ", article)
        #
        result.append(article.lower())

    return result

def keyword_in_sentences(thread_idx, data, keywords):
    """
    Sentence keyword search (called by threads or used normally)

    Arguments:
        thread_idx(int): Order of threads, used to align progressbar.
        data(list of str): Each elements in the list contains one sentence
                          of raw corpus.

    Returns:
        result(list of str): Each elements in the list contains one 
                     sentence with one or more keywords.
    """
    # global keywords
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    #
    for line in tqdm(data, position=thread_idx, desc=desc):
        # search for keywords
        found = False
        found_keyword = list()
        for itr in keywords:
            # Append keywords to the list
            if itr.lower() in line.lower():
                found = True
                found_keyword.append(itr)
        #
        if found:
            result.append(line + "\t" + "\t".join(found_keyword))

    return result
