import json, os, nltk
from tqdm import tqdm
from pprint import pprint
from utils import readlines, string_file_io, generic_threading


def parallel_gen_IOB(thread_idx, data):
    global entity
    desc = "Thread {:2d}".format(thread_idx + 1)
    final = list()

    for line in tqdm(data, position=thread_idx, desc=desc):
        sentence, entity_mention = line.split('\t')[0], line.split('\t')[1:]
        words = (nltk.word_tokenize(sentence))
        result = words
        for idx, word in enumerate(words):
            find = False
            for mention in entity_mention:
                entity_type = (entity[mention])
                for i in range(len(mention.split())):
                    if word.lower() == mention.split()[i].lower():
                        if i == 0:
                            result[idx] += '\tB-'+entity_type
                            find = True
                        else:
                            result[idx] += '\tI-'+entity_type
                            find = True
            if not find:
                result[idx] += '\tO'
        final.append(result)
    final = [itr for e in final for itr in e]
    return final

def main():
    # filename
    rule = "../data/"
    corpus = "../data/smaller_preprocessed_sentence_keywords.tsv"
    thread = 10

    # merge dictionary
    files = [itr for itr in os.listdir(rule) if itr.endswith("_leaf.json")]
    global entity
    entity = dict()
    for itr in files:
        entity.update(json.load(open(rule + itr, "r")))
    
    # read corpus
    raw_data = readlines(corpus, limit=None)
    result = generic_threading(thread, raw_data, parallel_gen_IOB)
    #print(result)
    string_file_io("output.tsv", result)

main()

