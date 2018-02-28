import multiprocessing
from multiprocessing import Pool, cpu_count
from pprint import pprint


def vprint(msg, verbose=False):
    """
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

def generic_threading(n_jobs, data, threading_method):
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
        result = p.starmap(threading_method, thread_data)
    #
    print("\n" * n_threads)
    print("All threads completed.")
    return result