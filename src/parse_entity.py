import re
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import merge_dict, multi_mentions, write_to_file, vpprint, vprint, readlines
from collections import Counter
from itertools import chain
from tqdm import tqdm
from pprint import pprint
from copy import copy
"""
python src/parse_entity.py data/MeSH_type_hierarchy.txt --trim --threshold=1
python src/parse_entity.py data/UMLS_type_hierarchy.txt --trim --threshold=1
python src/parse_entity.py data/custom_subwords_v2.txt --subword
"""


def entity_parser(file, trim=True, threshold=1, plot=False):
    """
    Args:
        file(str): Path to the hierarchy tree file.
        trim(bool): Trim the hierarchy tree.
        threshold(int): Threshold for trimming the hierarchy tree.
        plot(bool): Plot the occurence of the nodes.
    """
    # Parsing File
    dataset = readlines(file, begin=1, limit=None)
    keys = list()
    value = list()
    # Preliminary filling the hierarchy tree
    for data in dataset:
        # [hierarchy_index] [entity_name]
        temp = re.split(r"[\t]", data)
        keys.append(temp[0])
        value.append(temp[1])
    print("\nNumber of nodes in the hierarchy: {:3d}".format(len(keys)))

    # Initialize dictionary with keys as key and empty list as value
    entity = {k: [] for k in keys}
    print("Initializing hierarchy tree with indices...")
    total_len = len(keys)
    #
    leaf = list()
    depth = 0
    for idx, k in enumerate(keys):
        # Add root type
        entity[k].append(k[0])
        # hierarchy path
        path = k.split('.')
        # Append all indice along the path
        entity[k] += ['.'.join(path[:i + 1]) for i in range(len(path))]
        # Check leaf nodes
        if depth < len(entity[k]):
            # print(" - Update depth to {:d}".format(len(entity[k])))
            for itr_leaf in leaf:
                if keys[itr_leaf] not in k:
                    entity[keys[itr_leaf]].append('*')
                else:
                    continue
            # clear leaf list
            del leaf[:]
        # node of the same depth: just append to the leaf list
        elif depth == len(entity[k]):
            # print(" - Found same depth")
            pass
        # mark leaf nodes and clear leaf list
        else:
            # If the depth is larger than the current node
            # all the nodes in the list must be leaves
            # print(" - Leaves: {:s}".format([keys[i] for i in leaf]))
            # add indicator to the node
            for itr_leaf in leaf:
                entity[keys[itr_leaf]].append('*')
            # clear the leaf list
            del leaf[:]
        # update depth
        depth = len(entity[k])
        # add node to leaf list
        leaf.append(idx)
        # Final adjustment on last group of leaves
        if k == keys[-1]:
            for itr_leaf in leaf:
                entity[keys[itr_leaf]].append('*')
            # Clear up the list
            del leaf[:]
        else:
            continue

    print()
    print("Filling entity names...")
    # Filling the entity with there names according to the hierarchy
    for key in tqdm(entity):
        # Leaf node correspond to a entity
        if entity[key][-1] == '*':
            # copy the types
            type_list = [value[keys.index(itr)] for itr in entity[key][:-1]]
            # parsing the mentions (TBC)
            mention = type_list[-1]
            # print("{0}".format(mention))
            if ", " in mention:
                if ' or ' in mention:
                    # mention = mention.replace('or', '')
                    mention = mention.replace('or', ', ')
                # synonym = mention.split(',')
                mention = mention.split(', ')
                # print("*** {0} ***".format(mention + synonym))
            # no different mentions
            else:
                mention = [mention]
                pass
            # create a dictionary with types and mentions
            entity[key] = {
                "TYPE": type_list,
                "MENTION": mention,
                "PATH": [copy(type_list)]
            }
            # print(mention)
            pass
        else:
            for idx, element in enumerate(entity[key]):
                # lookup the hierarchy index in the key list
                # element: hierarchy index iterator
                index = keys.index(element)
                # Fill in the names
                entity[key][idx] = value[index]
    #
    def uni_list(arr):
        """
        Generate unique list
        """
        return list(np.unique(arr))

    print()
    print("Replacing key names and merge duplicated names...")
    # check duplicated key
    for i in tqdm(range(len(keys))):
        # Merge leaves
        name = value[i]
        if name in entity:
            duplicate = entity.pop(keys[i])
            #
            if type(entity[name]) == dict:
                if type(duplicate) == dict:
                    # print("found duplicate key in {:s}".format(name))
                    entity[name]["PATH"].append(copy(duplicate["TYPE"]))
                    entity[name]["TYPE"] += duplicate["TYPE"]
                    entity[name]["MENTION"] += duplicate["MENTION"]
                else:
                    entity[name]["TYPE"] += duplicate
                    # entity[name]["MENTION"] += duplicate
                # convert to unique list
                entity[name]["TYPE"] = uni_list(entity[name]["TYPE"])
                entity[name]["MENTION"] = uni_list(entity[name]["MENTION"])
            else:
                entity[name] += duplicate
                entity[name] = uni_list(entity[name])
        # Replace names
        else:
            entity[name] = entity.pop(keys[i])
    # pprint(entity)
    # Save
    save_name = file[:-4] + "_index.json"
    write_to_file(save_name, entity)

    print()
    print("Generating leaf node file...")
    leaf_info = dict()
    # Output only leaf node dictionary
    leaf_name = file[:-4] + "_leaf.json"
    invalid_list = list()
    for entry in tqdm(entity):
        # print(entry)
        if type(entity[entry]) == dict:
            synonym = multi_mentions(entry)
            for itr in synonym:
                # A, B -> A BA: if A is a node in the tree but not a leaf node
                # SKIP the synonym
                try:
                    if type(entity[itr]) is not dict:
                        # print("Found invalid: {0} | {1}".format(itr, entry))
                        invalid_list.append(itr)
                        continue
                except:
                    pass
                # leaf_info[itr] = entity[entry]["TYPE"]
                try:
                    leaf_info[itr]["TYPE"] += copy(entity[entry]["TYPE"])
                    leaf_info[itr]["PATH"] += copy(entity[entry]["PATH"])
                except:
                    leaf_info[itr] = dict()
                    leaf_info[itr]["TYPE"] = copy(entity[entry]["TYPE"])
                    leaf_info[itr]["PATH"] = copy(entity[entry]["PATH"])
        else:
            pass
    # pprint(leaf_info)
    if len(invalid_list) > 0:
        write_to_file(file[:-4] + "_invalid.txt", invalid_list)
    # Save leaf node file
    write_to_file(leaf_name, leaf_info)
    #
    print("Building k-parents tree")
    tmp_key = list(leaf_info.keys())
    tmp_val = [itr["PATH"] for itr in leaf_info.values()]
    c = [len(itr) for itr in chain.from_iterable(tmp_val)]
    print(" - Tree Depth: MIN={:2d} | MAX={:2d}".format(np.min(c), np.max(c)))
    kptree = dict(zip(tmp_key, tmp_val))
    write_to_file(file[:-4] + "_kptree.json", kptree)

    # Trim the hierarchy tree
    if trim and threshold > 0:
        print()
        print("Counting occurence of all labels...")
        # all_types = list(leaf_info.values())
        all_types = [itr["TYPE"] for itr in leaf_info.values()]
        all_types = list(chain.from_iterable(all_types))
        n_org_labels = len(uni_list(all_types))
        # print(n_org_labels)
        occurence = dict(Counter(all_types))
        # pprint(occurence)

        if plot:
            # print("Type frequencies and their corresponding amount:")
            frequency = list(occurence.values())
            statistics = dict(Counter(frequency))
            n_total_types = sum(list(statistics.values()))
            # pprint(statistics)

            x = list(statistics.keys())
            y = list(statistics.values())
            # sort x, y
            y = [y[itr] for itr in list(np.argsort(x))]
            x = sorted(x)

            if len(x) >= 100:
                x = x[:100]
                y[99] = sum(y[99:])
                y = y[:100]
            # normalize
            total_labels = sum(y)
            y = [100. * itr / total_labels for itr in y]
            y = list(np.cumsum(y))
            #
            plt.plot(x, y)
            plt.title("Cumulative label occurence (Total: {0})".format(
                n_total_types))
            plt.xlabel("occurence (occurence > 100 are treated as 100)")
            # plt.ylabel("# of labels")
            plt.ylabel("Percentage of all labels(%)")
            # plt.show()
            img_name = "{0}_occurence.png".format(file[:-4])
            plt.savefig(img_name, dpi=300)
            print(" - Statistics saved in {0}".format(img_name))

        # Removing infrequent labels
        print("Trimming infrequent labels with occurence threshold = {:3d}"
              .format(threshold))
        # trimmed_labels = list(leaf_info.values())
        trimmed_labels = [copy(itr["TYPE"]) for itr in leaf_info.values()]
        for itr_label, itr_occ in tqdm(occurence.items()):
            if itr_occ <= threshold:
                # print(" - Removing infrequent label: {0} (occurence = {1})"
                #        .format(itr_label, itr_occ))
                for idx in range(len(trimmed_labels)):
                    if itr_label in trimmed_labels[idx]:
                        trimmed_labels[idx].remove(itr_label)
            # skip if occurence > threshold
            else:
                pass
        trimmed_leaf = dict(zip(leaf_info.keys(), trimmed_labels))
        # Calculate some triming figures
        flatten_trim = list(chain.from_iterable(trimmed_labels))
        n_trim_labels = len(uni_list(flatten_trim))
        reduced = (n_org_labels - n_trim_labels)
        reduced_per = 100. * reduced / n_org_labels

        print("Trimmed labels from {:8d} to {:8d} (threshold = {:3d})".format(
            n_org_labels, n_trim_labels, threshold))
        print(" - Reduced labels by {:2.2f}% ({:5d} labels)".format(
            reduced_per, reduced))
        print(len(trimmed_leaf))

        # Save trimmed labels to file
        trimmed = file[:-4] + "_trimmed.json"
        write_to_file(trimmed, trimmed_leaf)
    else:
        print("Threshold = 0, no trimming.")


def extract_mentions(path, trim=True):
    """
    """
    entities = merge_dict(path, trim=trim)

    mentions = list(entities.keys())
    print(len(mentions))
    write_to_file("mention_list.txt", mentions)
    pass


def parse_subwords(file):
    """
    Parse subword mapping to dictionary.

    Args:
        file(str): Path to the subword mapping file.
                   format: <[WORD]>S1,S2,...
    """
    # Load subwords
    raw_data = readlines(file, limit=None)

    # Parsing information
    mentions, subwords = list(), list()
    for itr in raw_data:
        index = itr.find(">")
        mentions.append(itr[1:index])
        subwords.append(itr[index + 2:].split(","))

    # Create dictionary for mentions and its subwords
    dictionary = dict(zip(mentions, subwords))

    write_to_file("data/subwords.json", dictionary)

    # Subword pool for subword embedding
    # subword_pool = np.unique(list(chain.from_iterable(subwords)))
    subword_pool = dict(Counter(list(chain.from_iterable(subwords))))
    # print(subword_pool)
    print("Raw number of subwords: {:8d}".format(len(subword_pool)))
    write_to_file("data/subword_pool.json", subword_pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to be parsed.")
    parser.add_argument(
        "--subword", action="store_true", help="Parse subword list.")
    parser.add_argument(
        "--trim", action="store_true", help="Trim the hierarchy tree.")
    parser.add_argument(
        "--threshold",
        nargs='?',
        type=int,
        default=2,
        help="occurence below threshold would be filtered out.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot statistics if the argument is given.")

    args = parser.parse_args()

    if args.subword:
        parse_subwords(args.file)
    else:
        entity_parser(args.file, args.trim, args.threshold, args.plot)
