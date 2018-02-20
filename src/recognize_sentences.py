import os
import argparse
import json
from pprint import pprint
from mmap import mmap
from tqdm import tqdm

def get_num_lines(file_path):
    lines = 0
    with open(file_path, "r+") as fp:
        buf = mmap(fp.fileno(), 0)
        while buf.readline():
            lines +=1
    return lines

def recognize_sentences(args):
    total_lines = get_num_lines(args.sentences)
    with open(args.sentences, "r") as f_in, open(args.found, "w") as f_out:
        # load dictionarys
        files = [itr for itr in os.listdir(args.dict_path) 
                 if itr.endswith('_leaf.json')]
        #
        entity = dict()
        for itr in files:
            entity.update(json.load(open(args.dict_path + '/' + itr, "r")))
        #
        keywords = entity.keys()
        count = 0
        for line in tqdm(f_in, total=total_lines):
            # search for keywords
            for itr in keywords:
                if itr in line:
                    # print(" - Found sentence with keyword: {0}".format(itr))
                    f_out.write(line)
                    break
                else:
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences", help="Input sentences to be recognized.")
    parser.add_argument("found", help="Sentences with key words")
    parser.add_argument("dict_path", help="Put all dictionaries \
                         with extension \".json\".")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    recognize_sentences(args)