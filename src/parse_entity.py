import re
import pprint
import json
import numpy as np
import argparse


def entity_parser(args):

    def vprint(msg):
        if args.verbose:
            print(msg)

    def vpprint(msg):
        if args.verbose:
            pprint.pprint(msg)

    # parsing file
    with open(args.file, 'r') as f:
        dataset = f.read().splitlines()[1:]

        keys = list()
        value = list()
        for data in dataset:
            # split line: [hierarchy_index] [entity_name]
            temp = re.split(r"[\t]", data)
            keys.append(temp[0])
            value.append(temp[1])
        print("Length of keys {:3d}".format(len(keys)))

        # initialize dictionary with keys as key and empty list as value
        entity = {k: [] for k in keys}
        print("Start append...")
        total_len = len(keys)
        #
        leaf = list()
        depth = 0
        for idx, k in enumerate(keys):
            if idx % 100 == 0:
                vprint("step {:3d} / {:3d}".format(idx + 1, total_len))
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
        #
        vpprint(entity)
        # exit()

        print("Filling entity names...")
        # Filling the entity with there names according to the hierarchy
        for key in entity:
            # Leaf node correspond to a entity
            if entity[key][-1] == '*':
                # copy the types
                type_list = [value[keys.index(itr)]
                             for itr in entity[key][:-1]]
                # parsing the mentions (TBC)
                mention = type_list[-1]
                # print("{0}".format(mention))
                if ", " in mention:
                    if ' or ' in mention:
                        mention = mention.replace('or', '')
                    # synonym = mention.split(',')
                    mention = mention.split(', ')
                    # print("*** {0} ***".format(mention + synonym))
                # no different mentions
                else:
                    mention = [mention]
                    pass
                # create a dictionary with types and mentions
                entity[key] = {"TYPE": type_list, "MENTION": mention}
                # print(mention)
                pass
            else:
                for idx, element in enumerate(entity[key]):
                    # lookup the hierarchy index in the key list
                    # element: hierarchy index iterator
                    index = keys.index(element)
                    # fill in the names
                    entity[key][idx] = value[index]

        def uni_list(arr):
            """
            Generate unique list
            """
            return list(np.unique(arr))

        save_name = args.file[:-4] + "_index.json"
        print("Replacing key names and merge duplicated names...")
        # check duplicated key
        for i in range(len(keys)):
            # Merge leaves
            name = value[i]
            if name in entity:
                duplicate = entity.pop(keys[i])
                print(duplicate)
                print(entity[name])
                #
                if type(entity[name]) == dict:
                    if type(duplicate) == dict:
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

        vpprint(entity)
        # Save
        with open(save_name, 'w') as fp:
            json.dump(entity, fp, sort_keys=True, indent=4)
        print("File saved in {:s}".format(save_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to be parsed.")
    parser.add_argument("-o", "--output", help="Output file name, postfix \
                        \"_index\" would be added if this argument is not \
                        given. [file_type: .json]")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    entity_parser(args)
