import json
import os
import nltk
import argparse
from tqdm import tqdm
from utils import readlines, string_file_io, generic_threading

# python src/create_iob.py data/smaller_preprocessed_sentence_keywords.tsv data/keywords.json --thread=10

def generate_ibo(thread_idx, data, entity):
    """
    Generate typing according th IOB format.

    Arguments:
        thread_idx(int): Order of threads, used to align progressbar.
        data(list of str):Each elements in the list contains one sentence.
        entity(dict): Keywords dictionary with mentions as key and types
                      as values.
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    final = list()
    # IBO typing:
    # Priority: I > B > O
    type_I = "\tI-"
    type_B = "\tB-"
    type_O = "\tO"

    for line in tqdm(data, position=thread_idx, desc=desc):
        # Format: [sentence] \tab [mentions]
        sentence, entity_mentions = line.split('\t')[0], line.split('\t')[1:]
        # Decompose the sentence to word-level
        words = nltk.word_tokenize(sentence)
        len_sentence = len(words)
        # Duplicate the words for output
        result = words.copy()

        # Type O: Default type
        for i in range(len_sentence):
            result[i] += type_O

        # Iterate all entity mentions and append its type:
        # To prevent some entity mentions are subwords in others, we put
        # a sliding window with length equal to the size of the entity
        # mention in concern, and slide it through the sentence.
        #
        # Mentions: 
        # (1) University of Illinois
        # (2) Illinois
        # (3) student
        # 
        # Sentence: 
        # He is a student in University of Illinois at Urbana-Champaign.
        # O  O  O    B    O      B      I      I    O        O
        # 
        for mention in entity_mentions:
            # Add delimitor
            entity_type = "@ ".join(entity[mention])
            # Window length (number of words in the entity mention)
            len_window = len(mention.split())
            # Slide the window through the entire sentence
            # The head of the window slides from index 0 to (-window size)
            for begin in range(len_sentence - len_window):
                # print(words[begin:begin + len_window])
                # Reconstruct the words within the sliding window
                tmp = " ".join(words[begin:begin + len_window])
                # If the word has been mark as type I and is found to be
                # the mention in interest, then it must be the subword of
                # another mention that mark this word as type I.
                #
                # e.g. 
                # If the mention "University of Illinois" marked both "of"
                # and "Illinois" as type I. In some future iteration, the
                # mention "Illinois" is also found to be a valid word in
                # the sliding window, but since the word "Illinois" is the
                # subword of "University of Illinois", it should remain as
                # type I instead of remarking it as type B.
                #
                # If the mention is found in the sliding window
                if tmp.lower() == mention.lower():
                    # Avoid remarking type I words as type B words
                    if type_I in result[begin]:
                        continue
                    # Mark first word as type B and rests as type I
                    else:
                        typing = type_B + entity_type
                        # Replace type_O as type_B
                        result[begin] = result[begin].replace(type_O, typing)
                        # Mark rest of the words in the window as type I,
                        # Replace type_O as type_I
                        for j in range(1, len_window):
                            typing = type_I + entity_type
                            result[begin + j] = result[begin + j].replace(type_O, typing)
                #
                # The mention is not in the sliding window
                else:
                    pass
        final.append(result)

    final = [itr for e in final for itr in e]
    return final

def ibo_tagging(corpus, keywords, output=None, thread=None):
    """
    Arguments:
        corpus(str): Path to the corpus file.
        keywords(str): Path to where keywords dictionaries is.
        thread(int): Number of thread to process.
        output(str): Path to the output file.
    """
    # output name
    if output is None:
        output = corpus[:-4] + "_ibo.tsv"

    # Load and merge dictionary
    # files = [itr for itr in os.listdir(rule) if itr.endswith("_leaf.json")]

    # Load entities
    # entity = dict()
    # for itr in files:
    #     entity.update(json.load(open(rule + itr, "r")))
    entity = json.load(open(keywords, "r"))

    # Read corpus
    raw_data = readlines(corpus)

    # Threading
    param = (entity,)
    result = generic_threading(thread, raw_data, generate_ibo, param)

    # Write result to file
    string_file_io(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Input sentences to be recognized.")
    parser.add_argument("keywords", help="Path to the keyword dictionary.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores") 

    args = parser.parse_args()

    ibo_tagging(args.corpus, args.keywords, args.output, args.thread)
