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

# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --train_idx=./model/new_train_index.pkl --test_idx=./model/new_test_index.pkl
# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled_subwords.tsv --subword --train_idx=./model/new_train_index.pkl --test_idx=./model/new_test_index.pkl

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

def run(model_dir, input, test_size, train_idx, test_idx, subword=False):
    MAX_MENTION_LENGTH = 5 if not subword else 15
    print(MAX_MENTION_LENGTH)
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Create directory to store model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading dataset..")
    dataset = pd.read_csv(input, sep='\t', names=['label','context','mention'])

    X = dataset['context'].values
    y = dataset['label'].values
    mentions = dataset['mention'].values

    train_index = pkl.load(open(train_idx, 'rb'))
    test_index = pkl.load(open(test_idx, 'rb'))

    X_train, X_test = X[train_index], X[test_index]
    X_train_mention, X_test_mention = mentions[train_index], mentions[test_index]

    print("Writing new_test_mention_list..")
    with open(model_dir + "test_mention_list.txt", "w") as f:
        for mention in X_test_mention:
            f.write(mention + "\n")


    del X, mentions
    
    print("Tokenize sentences...")
    tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(list(X_train))
    list_tokenized_train = tokenizer.texts_to_sequences(X_train)
    list_tokenized_test = tokenizer.texts_to_sequences(X_test)
    del X_train, X_test
    # Padding sentences
    print("Padding sentences vector...")
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_SEQUENCE_LENGTH)
    del list_tokenized_train, list_tokenized_test

    if subword:
        pkl.dump(X_t, open(model_dir + "training_data_w_subword_filter.pkl", 'wb'))
        pkl.dump(X_te, open(model_dir + "testing_data_w_subword_filter.pkl", 'wb'))
    else:
        pkl.dump(X_t, open(model_dir + "training_data_wo_subword_filter.pkl", 'wb'))
        pkl.dump(X_te, open(model_dir + "testing_data_wo_subword_filter.pkl", 'wb'))
    del X_t, X_te

    print("Tokenize mentions...")
    m_tokenizer = text.Tokenizer(num_words=MAX_NUM_MENTION_WORDS)
    m_tokenizer.fit_on_texts(list(X_train_mention))
    m_list_tokenized_train = m_tokenizer.texts_to_sequences(X_train_mention)
    m_list_tokenized_test = m_tokenizer.texts_to_sequences(X_test_mention)
    del X_train_mention, X_test_mention

    # Padding mentions
    print("Padding mentions vector...")
    X_m_t = sequence.pad_sequences(m_list_tokenized_train, maxlen=MAX_MENTION_LENGTH)
    X_m_te = sequence.pad_sequences(m_list_tokenized_test, maxlen=MAX_MENTION_LENGTH)
    del m_list_tokenized_train, m_list_tokenized_test

    if subword:
        pkl.dump(X_m_t, open(model_dir + "training_mention_w_subword_filter.pkl", 'wb'))
        pkl.dump(X_m_te, open(model_dir + "testing_mention_w_subword_filter.pkl", 'wb'))
    else :
        pkl.dump(X_m_t, open(model_dir + "training_mention_wo_subword_filter.pkl", 'wb'))
        pkl.dump(X_m_te, open(model_dir + "testing_mention_wo_subword_filter.pkl", 'wb'))
    del X_m_t, X_m_te

    
    # Parsing the labels and convert to integer using comma as separetor
    print("Creating MultiLabel..")
    temp = list()
    for element in y:
        values = element.split(',')
        values = list(map(int, values))
        temp.append(values)
    # Convert to np.array
    del y
    temp = np.array(temp)
    print(len(temp), len(temp[0]))
    print(type(temp), type(temp[0]))
    y_train = temp[train_index]
    y_test = temp[test_index]
    # Binarizer the labels
    print("Binarizering labels..")
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(temp)
    del temp

    y_train = mlb.transform(y_train)
    y_test = mlb.transform(y_test)
    print(" shape of training labels:",y_train.shape)
    print(" shape of testing labels:",y_test.shape)

    # dumping training and testing label
    if subword:
        pkl.dump(y_train, open(model_dir + "training_label_w_subword_filter.pkl", 'wb'))
        pkl.dump(y_test, open(model_dir + "testing_label_w_subword_filter.pkl", 'wb'))
    else:
        pkl.dump(y_train, open(model_dir + "training_label_wo_subword_filter.pkl", 'wb'))
        pkl.dump(y_test, open(model_dir + "testing_label_wo_subword_filter.pkl", 'wb'))
    del y_train, y_test

    print("dumping pickle file of tokenizer/m_tokenizer/mlb...")
    # dumping model
    if subword:
        pkl.dump(tokenizer, open(model_dir + "tokenizer_w_subword_filter.pkl", 'wb'))
        pkl.dump(m_tokenizer, open(model_dir + "m_tokenizer_w_subword_filter.pkl", 'wb'))
        pkl.dump(mlb, open(model_dir + "mlb_w_subword_filter.pkl", 'wb'))
    else:
        pkl.dump(tokenizer, open(model_dir + "tokenizer_wo_subword_filter.pkl", 'wb'))
        pkl.dump(m_tokenizer, open(model_dir + "m_tokenizer_wo_subword_filter.pkl", 'wb'))
        pkl.dump(mlb, open(model_dir + "mlb_wo_subword_filter.pkl", 'wb'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs='?', type=str, default="model/", 
                        help="Directory to store models. [Default: \"model/\"]")
    parser.add_argument("--input", help="Input dataset filename.")
    parser.add_argument("--train_idx", help="Input training index pickle file")
    parser.add_argument("--test_idx", help="Input testing index pickle file")
    parser.add_argument("--test_size", nargs='?', const=0.1, type=float, default=0.1,
                        help="Specify the portion of the testing data to be split.\
                        [Default: 10\% of the entire dataset]")
    parser.add_argument("--subword", action="store_true" , help="Use subword or not")
    args = parser.parse_args()

    run(args.model, args.input, args.test_size, args.train_idx, args.test_idx, args.subword)

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
