import pandas as pd
import numpy as np
import argparse
import itertools
from tqdm import tqdm
import pickle as pkl
from utils import generic_threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import text, sequence
import os

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

def run(model_dir, input, subword=False, vector=True):
    MAX_MENTION_LENGTH = 5 if not subword else 15
    print("MAX_MENTION_LENGTH = {0}".format(MAX_MENTION_LENGTH))
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Create directory to store model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    sb_tag = "w" if subword else "wo"

    print("Loading dataset..")
    dataset = pd.read_csv(input, sep='\t', names=['label','context','mention', 'subword'])

    X = dataset['context'].values
    y = dataset['label'].values
    mentions = dataset['mention'].values
    subwords = dataset['subword'].values

    # Parsing the labels and convert to integer using comma as separetor
    print("Creating MultiLabel Binarizer..")
    temp = list()
    for element in y:
        values = [int(itr) for itr in element.split(',')]
        # values = list(map(int, values))
        temp.append(values)
    del y
    # Convert to np.array
    temp = np.array(temp)

    # Parse subwords
    subwords = [str(itr).split(" ") for itr in subwords]
    ### Load subword pool
    ### Choose criteria to only include useful subwords
    ### Choose vector dimension

    # Initialize content tokenizer
    X_tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
    m_tokenizer = text.Tokenizer(num_words=MAX_NUM_MENTION_WORDS)
    # Fit MLB
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(temp)

    partitions = ["train", "test", "validation"]
    for itr in partitions:
        prefix = itr + "ing" if itr in ["train", "test"] else itr
        # Load designated indices for each partitions
        indices = pkl.load(open(model_dir + itr + "_index.pkl", 'rb'))
        # Index the content according to the given indices
        X_itr = X[indices]
        m_itr = mentions[indices]

        # Tokenization on the context
        print("Tokenize {0} sentences and mentions...".format(prefix))
        # Trim the token size w.r.t training context
        if itr == "train":
            X_tokenizer.fit_on_texts(list(X_itr))
            m_tokenizer.fit_on_texts(list(m_itr))
        # Tokenize the current context
        X_tokenized = X_tokenizer.texts_to_sequences(X_itr)
        m_tokenized = m_tokenizer.texts_to_sequences(m_itr)

        # Padding contexts
        print("Padding {0} sentences and mention vectors...".format(prefix))
        X_pad = sequence.pad_sequences(X_tokenized, maxlen=MAX_SEQUENCE_LENGTH)
        m_pad = sequence.pad_sequences(m_tokenized, maxlen=MAX_MENTION_LENGTH)

        # Save context vectors to pickle file
        # Sentence
        filename = "{0}{1}_data_{2}_subword_filter.pkl".format(model_dir, prefix, sb_tag)
        pkl.dump(X_pad, open(filename, 'wb'))
        # Mention
        filename = "{0}{1}_mention_{2}_subword_filter.pkl".format(model_dir, prefix, sb_tag)
        pkl.dump(m_pad, open(filename, 'wb'))
        del X_itr, X_tokenized, X_pad, m_itr, m_tokenized, m_pad

        # Binarizer the labels
        print("Binarizering labels..")
        y_itr = temp[indices]
        y_bin = mlb.transform(y_itr)
        print(" - {0} label shape: {1}".format(prefix, y_bin.shape))

        # Save label vectors to pickle file
        filename =  "{0}{1}_label_{2}_subword_filter1.pkl".format(model_dir, prefix, sb_tag)
        pkl.dump(y_bin, open(filename, 'wb'))

    # Save all models
    print("dumping pickle file of tokenizer/m_tokenizer/mlb...")
    pkl.dump(X_tokenizer, open(model_dir + "tokenizer_{0}_subword_filter.pkl".format(sb_tag), 'wb'))
    pkl.dump(m_tokenizer, open(model_dir + "m_tokenizer_{0}_subword_filter.pkl".format(sb_tag), 'wb'))
    pkl.dump(mlb, open(model_dir + "mlb_{0}_subword_filter.pkl".format(sb_tag), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs='?', type=str, default="model/", 
                        help="Directory to store models. [Default: \"model/\"]")
    parser.add_argument("--input", help="Input dataset filename.")
    # parser.add_argument("--train_idx", help="Input training index pickle file")
    # parser.add_argument("--test_idx", help="Input testing index pickle file")
    # parser.add_argument("--vali_idx", help="Input validation index pickle file")
    parser.add_argument("--subword", action="store_true" , help="Use subword or not")
    parser.add_argument("--vector", action="store_false" , help="Use vector-based subword information.")
    args = parser.parse_args()

    run(args.model, args.input, args.subword, args.vector)

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
