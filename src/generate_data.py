import pandas as pd
import numpy as np
import argparse
import itertools
from tqdm import tqdm
import pickle as pkl
from utils import generic_threading, readlines
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import text, sequence
import os

# python ./src/generate_data.py --input=../share/data_labeled_kpb.tsv --tag=kbp
# python ./src/generate_data.py --input=../share/kbp_ascii_labeled_kpb.tsv --tag=kbp
# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv
# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled_subwords.tsv --subword

# Feature-parameter..
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5
EMBEDDING_DIM = 100

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


def run(model_dir, input, subword=False, tag=None, vector=True):
    postfix = ("_" + tag) if tag is not None else ""
    MAX_MENTION_LENGTH = 5 if not subword else 15
    print("MAX_MENTION_LENGTH = {0}".format(MAX_MENTION_LENGTH))
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Create directory to store model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    sb_tag = "w" if subword else "wo"

    print("Loading dataset from: {:s}".format(input))
    cols = ['label', 'context', 'mention', 'begin', 'end']

    dataset = readlines(input, delimitor="\t")
    dataset = pd.DataFrame(dataset, columns=cols, dtype=str)
    """
    dataset = pd.read_csv(input, sep='\t', names=cols)
    """
    dataset['label'] = dataset['label'].astype(str)
    dataset['mention'] = dataset['mention'].astype(str)

    X = dataset['context'].values
    mentions = dataset['mention'].values
    # subwords = dataset['subword'].values

    # Parsing the labels and convert to integer using comma as separetor
    ##################################################
    """
    print(dataset.shape)
    print(dataset.ix[57514])
    print()
    print(dataset.ix[57514]['context'])
    print()
    print(dataset.ix[57514]['mention'])
    print()
    print(dataset.ix[57514]['begin'])
    print()
    print(dataset.ix[57514]['end'])
    """
    ##################################################
    """
    for idx, itr in enumerate(dataset['begin'].values):
        print(idx, [int(e) for e in itr.split(',')])
    """
    y = np.array(
        [[int(itr) for itr in e.split(',')] for e in dataset['label'].values])
    b_position = [[int(itr) for itr in element.split(',')]
                  for element in dataset['begin'].values]
    b_position = np.array(b_position)
    e_position = [[int(itr) for itr in element.split(',')]
                  for element in dataset['end'].values]
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

    partitions = ["train", "test", "validation"]
    for itr in partitions:
        prefix = itr + "ing" if itr in ["train", "test"] else itr
        # Load designated indices for each partitions
        filename = model_dir + itr + "_index{:s}.pkl".format(postfix)
        print("Loading indices from file: {:s}".format(filename))
        indices = pkl.load(open(filename, 'rb'))
        # Index the content according to the given indices
        X_itr = X[indices]
        m_itr = mentions[indices]
        b_itr = b_position[indices]
        e_itr = e_position[indices]

        # Tokenization on the context
        print("Tokenize {0} sentences and mentions...".format(prefix))
        # Trim the token size w.r.t training context
        if itr == "train":
            X_tokenizer.fit_on_texts(list(X_itr))
            m_tokenizer.fit_on_texts(list(m_itr))

        # Tokenize the current context
        X_tokenized = X_tokenizer.texts_to_sequences(X_itr)
        m_tokenized = m_tokenizer.texts_to_sequences(m_itr)

        ######################################################################
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
        max_len, max_idx = max(length), np.argmax(length)
        print(" - MAX: {} at {} ({}:{})".format(
            max_len, max_idx, b_position[max_len], e_position[max_len]))
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
        exit()
        ######################################################################
        # Padding contexts
        print("Padding {0} sentences and mention vectors...".format(prefix))
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
        indicator = np.empty(X_pad.shape)
        for idx, (b, e) in enumerate(zip(b_itr, e_itr)):
            print("positions: ", b, e)
            if len(b) > 1:
                exit()
                print(b, e)
                for idx in range(len(b) - 1):
                    b0 = b[idx]
                    e0 = e[idx]
                    b1 = b[idx + 1]
                    e1 = e[idx + 1]
                    if idx == 0:
                        indicator[idx, b0:e0 + 1] = 2
                        indicator[idx, :b0] = 1
                    # Calculate the number of words between two mention
                    in_between = b1 - e0 - 1
                    # [b0 e0] LLL RRR [b1 e1]
                    # Mention
                    indicator[idx, b1:e1 + 1] = 2
                    # to the Left
                    l_idx = (e0 + 1)
                    r_idx = in_between // 2 + 1
                    indicator[idx, l_idx:r_idx] = 1
                    # to the Right
                    l_idx = (e0 + 1) + in_between // 2
                    r_idx = b1
                    indicator[idx, l_idx:r_idx] = 3
                    pass

            else:
                bb = b[0]  # - MAX_SEQUENCE_LENGTH
                ee = e[0]  # - MAX_SEQUENCE_LENGTH
                # Mention: 2
                indicator[idx, bb:ee + 1] = 2
                # Left: 1
                indicator[idx, :bb] = 1
                # Right: 3
                indicator[idx, ee + 1:] = 3

            # Mark padding as zero
            padded_idx = np.where(X_pad[idx, :] == 0)[0]
            indicator[idx, padded_idx] = 0
            ############################################
            print(X_pad[idx, ])
            print(indicator[idx, ])
            print()
            ############################################
            # exit()

        exit()
        # Save context vectors to pickle file
        # Sentence
        filename = "{:s}{:s}_data_{:s}_subword{:s}.pkl".format(
            model_dir, prefix, sb_tag, postfix)
        pkl.dump(X_pad, open(filename, 'wb'))
        # Mention
        filename = "{:s}{:s}_mention_{:s}_subword{:s}.pkl".format(
            model_dir, prefix, sb_tag, postfix)
        pkl.dump(m_pad, open(filename, 'wb'))
        del X_itr, X_tokenized, X_pad, m_itr, m_tokenized, m_pad

        # Binarizer the labels
        print("Binarizering labels..")
        y_itr = temp[indices]
        y_bin = mlb.transform(y_itr)
        print(" - {0} label shape: {1}".format(prefix, y_bin.shape))

        # Save label vectors to pickle file
        filename = "{:s}{:s}_label_{:s}_subword{:s}.pkl".format(
            model_dir, prefix, sb_tag, postfix)
        pkl.dump(y_bin, open(filename, 'wb'))

    # Save all models
    print("Dumping pickle file of tokenizer/m_tokenizer/mlb...")
    pkl.dump(
        X_tokenizer,
        open(
            model_dir + "tokenizer_{:s}_subword{:s}.pkl".format(
                sb_tag, postfix), 'wb'))
    pkl.dump(
        m_tokenizer,
        open(
            model_dir + "m_tokenizer_{:s}_subword{:s}.pkl".format(
                sb_tag, postfix), 'wb'))
    pkl.dump(
        mlb,
        open(model_dir + "mlb_{:s}_subword{:s}.pkl".format(sb_tag, postfix),
             'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs='?',
        type=str,
        default="model/",
        help="Directory to store models. [Default: \"model/\"]")
    parser.add_argument("--input", help="Input dataset filename.")
    # parser.add_argument("--train_idx", help="Input training index pickle file")
    # parser.add_argument("--test_idx", help="Input testing index pickle file")
    # parser.add_argument("--vali_idx", help="Input validation index pickle file")
    parser.add_argument(
        "--subword", action="store_true", help="Use subword or not")
    parser.add_argument("--tag", type=str, help="Make tags on the files.")
    parser.add_argument(
        "--vector",
        action="store_false",
        help="Use vector-based subword information.")
    args = parser.parse_args()

    run(args.model, args.input, args.subword, args.tag, args.vector)
    ''' use for spliting data with mention specific 
    print("{0} unique mentions...".format(len(set(mentions))))
    unique, counts = np.unique(mentions, return_counts=True)
    mention_count = dict(zip(unique, counts))
    #mention_index = list()

    # need parallel
    print("processing mention_index...")
    param = (mentions, )
    key_list = list(mention_count.keys())
    # [['mention1',[idxes]],['mention2',[idxes]],...]
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
    pkl.dump(train_index, open(model_dir + "train_index.pkl", 'wb'))
    pkl.dump(test_index, open(model_dir + "test_index.pkl", 'wb'))
    '''
    ''' use for filter out error mention
    print("Loading error mention...")
    error_mention = pkl.load(open("error_mention.pkl", 'rb'))

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

    pkl.dump(train_index, open("model/new_train_index.pkl", 'wb'))
    pkl.dump(test_index, open("model/new_test_index.pkl", 'wb'))
    exit()
    '''
