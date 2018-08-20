import pandas as pd
import numpy as np
import argparse
import itertools
from tqdm import tqdm
import pickle as pkl
from utils import generic_threading, readlines
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import text, sequence
import os

# python ./src/generate_data.py --input=../share/data_labeled_kpb.tsv --tag=kbp
# python ./src/generate_data.py --input=../share/kbp_ascii_labeled_kpb.tsv --tag=kbp
# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv
# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled_subwords.tsv --use_subword

# Feature-parameter..
MAX_NUM_WORDS = 100000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
MAX_MENTION_LENGTH = 5

np.random.seed(0)


def parallel_index(thread_idx, mention_count, mentions):
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    for key in tqdm(mention_count, position=thread_idx, desc=desc):
        index = np.where(mentions == key)[0]
        temp = [key]
        temp.append(index.tolist())
        result.append(temp)

    return result


def run(model_dir, input, use_subword=False, tag=None, vector=True):
    #
    postfix = "{:s}{:s}".format("_subword" if use_subword else "",
                                ("_" + tag) if tag is not None else "")
    #
    MAX_MENTION_LENGTH = 5 if not use_subword else 15
    print("MAX_MENTION_LENGTH = {0}".format(MAX_MENTION_LENGTH))
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Create directory to store model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading dataset from: {:s}".format(input))
    cols = ["label", "context", "mention", "begin", "end"]

    dataset = readlines(input, delimitor="\t")
    dataset = pd.DataFrame(dataset, columns=cols, dtype=str)
    """
    dataset = pd.read_csv(input, sep="\t", names=cols)
    """
    dataset["label"] = dataset["label"].astype(str)
    dataset["mention"] = dataset["mention"].astype(str)

    X = dataset["context"].values
    mentions = dataset["mention"].values
    # subwords = dataset["subword"].values

    # Parsing the labels and convert to integer using comma as separetor
    y = np.array(
        [[int(itr) for itr in e.split(",")] for e in dataset["label"].values])
    b_position = [[int(itr) for itr in element.split(",")]
                  for element in dataset["begin"].values]
    b_position = np.array(b_position)
    e_position = [[int(itr) for itr in element.split(",")]
                  for element in dataset["end"].values]
    e_position = np.array(e_position)

    # Parse subwords
    # subwords = [str(itr).split(" ") for itr in subwords]
    ### Load subword pool
    ### Choose criteria to only include useful subwords
    ### Choose vector dimension

    print("Creating MultiLabel Binarizer...")
    # Initialize content tokenizer
    X_tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
    m_tokenizer = text.Tokenizer(num_words=MAX_NUM_MENTION_WORDS)
    # Fit MLB
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(y)

    partitions = ["training", "testing", "validation"]
    for itr in partitions:
        # Parse prefix
        prefix = model_dir + itr
        # Load designated indices for each partitions
        filename = "{:s}_index{:s}.pkl".format(prefix, postfix)
        print("Loading indices from file: {:s}".format(filename))
        indices = pkl.load(open(filename, "rb"))
        # Index the content according to the given indices
        X_itr = X[indices]
        m_itr = mentions[indices]
        b_itr = b_position[indices]
        e_itr = e_position[indices]

        # Tokenization on the context
        print("Tokenize {0} sentences and mentions...".format(itr))
        # Trim the token size w.r.t training context
        if itr == "training":
            print(" - Fitting tokenizers on training data.\n")
            X_tokenizer.fit_on_texts(list(X_itr))
            m_tokenizer.fit_on_texts(list(m_itr))

        # Tokenize the current context
        X_tokenized = X_tokenizer.texts_to_sequences(X_itr)
        m_tokenized = m_tokenizer.texts_to_sequences(m_itr)

        ######################################################################
        """
        # For debugging and choosing MAX_SEQUENCE_LENGTH
        length = [len(itr) for itr in X_tokenized]
        _gt = sum([itr > MAX_SEQUENCE_LENGTH for itr in length])
        _eq = sum([itr == MAX_SEQUENCE_LENGTH for itr in length])
        _lt = sum([itr < MAX_SEQUENCE_LENGTH for itr in length])
        print(" * MAX_SEQUENCE_LENGTH = {}".format(MAX_SEQUENCE_LENGTH))
        print(" - GT: {:d} ({:2.2f}%)".format(_gt,
                                              100. * _gt / len(X_tokenized)))
        print(" - EQ: {:d} ({:2.2f}%)".format(_eq,
                                              100. * _eq / len(X_tokenized)))
        print(" - LT: {:d} ({:2.2f}%)".format(_lt,
                                              100. * _lt / len(X_tokenized)))
        # Check for extreme
        # max_len, max_idx = max(length), np.argmax(length)
        # print(" - MAX: {} at {} ({}:{})".format(
        #     max_len, max_idx, b_itr[max_idx], e_itr[max_idx]))
        # Check how many mentions will be truncated

        b_info = [bb for itr in b_itr for bb in itr]
        e_info = [ee for itr in e_itr for ee in itr]
        # Out-of-Range (OOR)
        b_oor = sum([bb > MAX_SEQUENCE_LENGTH for bb in b_info])
        e_oor = sum([ee > MAX_SEQUENCE_LENGTH for ee in e_info])
        print(
            "\n * Mention indices Out-of-Seq (OOS) [B: begin, E: end, P: Part of mention]"
        )
        print(" - B : {:d} ({:2.2f}%)".format(b_oor,
                                              100. * b_oor / len(X_tokenized)))
        print(" - E : {:d} ({:2.2f}%)".format(e_oor,
                                              100. * e_oor / len(X_tokenized)))

        partial = (np.array(b_info) > MAX_SEQUENCE_LENGTH) * (
            np.array(e_info) > MAX_SEQUENCE_LENGTH)
        partial = partial.sum()
        print(" - P : {:d} ({:2.2f}%)".format(
            partial, 100. * partial / len(X_tokenized)))
        print(" - MAX_B: {}:{}".format(max(b_info), e_info[np.argmax(b_info)]))
        print(" - MAX_E: {}:{}".format(b_info[np.argmax(e_info)], max(e_info)))
        # Suggestions
        print("\n * Suggestions:")
        for itr in range(1, 20):
            current_max = MAX_SEQUENCE_LENGTH + 10 * itr
            b_tr = sum([bb > current_max for bb in b_info])
            e_tr = sum([ee > current_max for ee in e_info])
            print(" - Raise to {}".format(current_max))
            print(" - B : {:d} ({:2.2f}%)".format(
                b_tr, 100. * b_tr / len(X_tokenized)))
            print(" - E : {:d} ({:2.2f}%)".format(
                e_tr, 100. * e_tr / len(X_tokenized)))

            partial = (np.array(b_info) > current_max) * (np.array(e_info) >
                                                          current_max)
            partial = partial.sum()
            print(" - P : {:d} ({:2.2f}%)".format(
                partial, 100. * partial / len(X_tokenized)))
        """
        ######################################################################
        # Padding contexts
        print("Padding {:s} sentences and mention vectors...".format(itr))
        """
            Fill sequence with -1 to indicate true 0 instead of messing up
        with the first word (index = 0) given by tokenizer.
        """
        X_pad = sequence.pad_sequences(
            X_tokenized,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post")

        m_pad = sequence.pad_sequences(
            m_tokenized,
            maxlen=MAX_MENTION_LENGTH,
            padding="post",
            truncating="post")

        # Add indicator
        count, old_count = 0, 0
        indicator = np.empty(X_pad.shape)
        for idx, (b, e) in enumerate(zip(b_itr, e_itr)):
            #print("positions: ", b, e)
            if len(b) > 1:
                count += 1
                # Look one ahead, e.g. P1, P2 are positions, the loop run
                # just once, current P1, look ahead at P2
                for mini_idx in range(len(b) - 1):
                    # P1
                    b0, e0 = b[mini_idx], e[mini_idx]
                    # P2
                    b1, e1 = b[mini_idx + 1], e[mini_idx + 1]
                    if mini_idx == 0:
                        # Fill in mention indicator for the first occurence
                        indicator[idx, b0:e0 + 1] = 2
                        # Fill in Left before thr first occurence
                        indicator[idx, :b0] = 1
                    else:
                        pass

                    # Fill in L/R for words in-between the two consecutive
                    # occurence. First half for R second half for L. Tie
                    # breaking (for odd number of in-between words): more L
                    # e.g. [b0 e0] RRR LLL [b1 e1]  (in_between = 6)
                    # e.g. [b0 e0] RRR LLLL [b1 e1] (in_between = 7)
                    # Calculate the number of words between two mention
                    in_between = b1 - e0 - 1
                    # Mention
                    indicator[idx, b1:e1 + 1] = 2
                    # to the Right
                    # FROM: current end + 1
                    # TILL: current end + 1 + half length of in-between words
                    l_idx = (e0 + 1)
                    r_idx = (e0 + 1) + in_between // 2 + 1
                    indicator[idx, l_idx:r_idx] = 3
                    # to the Left
                    # FROM: current end + half length of in-between words
                    # TILL: next begin
                    l_idx = (e0 + 1) + in_between // 2
                    r_idx = b1
                    indicator[idx, l_idx:r_idx] = 1
                    pass

            else:
                bb = b[0]
                ee = e[0]
                # Mention: 2
                indicator[idx, bb:ee + 1] = 2
                # Left: 1
                indicator[idx, :bb] = 1
                # Right: 3
                indicator[idx, ee + 1:] = 3

            # Mark padding as zero
            # padded_idx = np.where(X_pad[idx, :] == -1)[0]
            padded_idx = np.where(X_pad[idx, :] == 0)[0]
            #indicator[idx, padded_idx] = 0
            indicator[idx, len(X_itr[idx].split(" ")):] = 0
            #if count > old_count:
            #print(X_itr[idx])
            #print(len(X_itr[idx].split(" ")))
            #print(m_itr[idx])
            #print(X_pad[idx, ])
            #print(indicator[idx, ])
            #print()
            """
                Fill padded positions with 0, i.e. fill indicator with zero
            after the end of sequence (index after length_of_sentence.)

            ISSUE: If array "indicator" is initialize as empty array, some
                Out-of-Vocabulary (OOV) words at the end of the sequence
                would remain as padded values.
            """
            # X_pad[idx, len(X_itr[idx].split(" ")):] = 0
            # print(X_pad[idx, ])
            ############################################

            ############################################
            #if count > 5:
            #    exit()
            #old_count = count
        """
        indicator_enc = OneHotEncoder()
        indicator = indicator[:, :, np.newaxis]
        indicator_enc.fit(indicator[0, :])
        indicator = np.array([indicator_enc.transform(tmp) for tmp in indicator])
        """
        # print("show indicator example", indicator[0], indicator.shape)

        # Save context vectors to pickle file
        # Sentence
        filename = "{:s}_context{:s}.pkl".format(prefix, postfix)
        pkl.dump(X_pad, open(filename, "wb"))
        print(" * Save context to {:s}".format(filename))
        # Mention
        filename = "{:s}_mention{:s}.pkl".format(prefix, postfix)
        pkl.dump(m_pad, open(filename, "wb"))
        print(" * Save mention to {:s}".format(filename))
        # Indicator
        filename = "{:s}_indicator{:s}.pkl".format(prefix, postfix)
        pkl.dump(indicator, open(filename, "wb"))
        print(" * Save indicator to {:s}".format(filename))

        del X_itr, X_tokenized, X_pad, m_itr, m_tokenized, m_pad, indicator

        # Binarizer the labels
        print("Binarizering labels..")
        y_itr = y[indices]
        y_bin = mlb.transform(y_itr)
        print(" - {0} label shape: {1}".format(prefix, y_bin.shape))

        # Save label vectors to pickle file
        filename = "{:s}_label{:s}.pkl".format(prefix, postfix)
        pkl.dump(y_bin, open(filename, "wb"))
        print(" * Save binarizered labels to {:s}\n".format(filename))

    # Save all models
    print("\nDumping pickle file of X_tokenizer/m_tokenizer/mlb...")
    filename = model_dir + "X_tokenizer{:s}.pkl".format(postfix)
    pkl.dump(X_tokenizer, open(filename, "wb"))
    print(" * Save X_tokenizer to {:s}".format(filename))

    filename = model_dir + "m_tokenizer{:s}.pkl".format(postfix)
    pkl.dump(m_tokenizer, open(filename, "wb"))
    print(" * Save m_tokenizer to {:s}".format(filename))

    filename = model_dir + "mlb{:s}.pkl".format(postfix)
    pkl.dump(mlb, open(filename, "wb"))
    print(" * Save mlb to {:s}".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        default="model/",
        help="Directory to store models. [Default: \"model/\"]")
    parser.add_argument("--input", help="Input dataset filename.")
    # parser.add_argument("--train_idx", help="Input training index pickle file")
    # parser.add_argument("--test_idx", help="Input testing index pickle file")
    # parser.add_argument("--vali_idx", help="Input validation index pickle file")
    parser.add_argument(
        "--use_subword", action="store_true", help="Use subword or not")
    parser.add_argument("--tag", type=str, help="Make tags on the files.")
    parser.add_argument(
        "--vector",
        action="store_false",
        help="Use vector-based subword information.")
    args = parser.parse_args()

    run(args.model, args.input, args.use_subword, args.tag, args.vector)
    """ use for spliting data with mention specific 
    print("{0} unique mentions...".format(len(set(mentions))))
    unique, counts = np.unique(mentions, return_counts=True)
    mention_count = dict(zip(unique, counts))
    #mention_index = list()

    # need parallel
    print("processing mention_index...")
    param = (mentions, )
    key_list = list(mention_count.keys())
    # [["mention1",[idxes]],["mention2",[idxes]],...]
    mention_index = generic_threading(20, key_list, parallel_index, param) 
    mention = []
    indices = []

    for metion_pair_thread in mention_index:
        for metion_pair in metion_pair_thread:
            mention.append(metion_pair[0])
            indices.append(metion_pair[1])

    mention_index = dict(zip(mention, indices))
    #mention_index = dict(zip(unique, mention_index))

    total_length = mentions.shape[0]
    test_len     = total_length * test_size
    train_len    = total_length - test_len
    train_index  = []
    count = 0
    print("processing training_index...")
    print("training size: {0}, testing size: {1}, total size: {2}".format(train_len, test_len, total_length))
    for mention in tqdm(key_list):
        count += mention_count[mention]
        if count < train_len:
            train_index.append(mention_index[mention])

    # flatten list
    print("flatten train_index...")
    dataset_index = set([i for i in range(total_length)])
    train_index = set(list(itertools.chain.from_iterable(train_index)))
    test_index = dataset_index - train_index  # use set property to extract index of testing
    print("train size:",len(train_index))
    train_index = np.array(list(train_index)) # transfer back to numpy array for further index
    test_index = np.array(list(test_index))   # transfer back to numpy array for further index

    # shuffle the index
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)
    print("train_index:",train_index)
    print("test_index:",test_index)
    pkl.dump(train_index, open(model_dir + "train_index.pkl", "wb"))
    pkl.dump(test_index, open(model_dir + "test_index.pkl", "wb"))
    """
    """ use for filter out error mention
    print("Loading error mention...")
    error_mention = pkl.load(open("error_mention.pkl", "rb"))

    count = 0
    train_error_idx = list()
    for idx, mention in enumerate(X_train_mention):
        if mention in error_mention:
            count+=1
            train_error_idx.append(idx)
            #print("error mention!")
    train_error_idx = np.array(train_error_idx)
    train_index = np.delete(train_index, train_error_idx)
    print("train error sentences! num:{:d}".format(count))

    count = 0
    test_error_idx = list()
    for idx, mention in enumerate(X_test_mention):
        if mention in error_mention:
            count+=1
            test_error_idx.append(idx)
    print("test error sentences! num:{:d}".format(count))
    test_error_idx = np.array(test_error_idx)
    test_index = np.delete(test_index, test_error_idx)

    pkl.dump(train_index, open("model/new_train_index.pkl", "wb"))
    pkl.dump(test_index, open("model/new_test_index.pkl", "wb"))
    exit()
    """
