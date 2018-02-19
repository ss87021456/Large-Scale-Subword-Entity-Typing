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
        for idx, k in enumerate(keys):
            if idx % 100 == 0:
                vprint("step {:3d} / {:3d}".format(idx + 1, total_len))
            # Add root type
            entity[k].append(k[0])
            # hierarchy path
            path = k.split('.')
            # Append all indice along the path
            entity[k] += ['.'.join(path[:i + 1]) for i in range(len(path))]

        print("Filling entity names...")
        # Filling the entity with there names according to the hierarchy
        for key in entity:
            for idx, element in enumerate(entity[key]):
                # lookup the hierarchy index in the key list
                # element: hierarchy index iterator
                index = keys.index(element)
                # fill in the names
                entity[key][idx] = value[index]

        save_name = args.file[:-4] + "_index.json"
        print("Replacing key names...")
        for i in range(len(keys)):
            # check duplicated key
            # if entity.has_key(value[i]):
            if value[i] in entity:
                entity[value[i]] = entity[value[i]] + entity.pop(keys[i])
                entity[value[i]] = list(np.unique(entity[value[i]]))
            else:
                entity[value[i]] = entity.pop(keys[i])

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
