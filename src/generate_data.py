import pandas as pd
import numpy as np
import argparse
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing import text, sequence
import os

# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5
EMBEDDING_DIM = 100

def run(model_dir, input, testing):
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
    mentions = dataset['label'].values

    X = np.array([(a, b) for a, b in zip(X, mentions)]) # create a structure numpy contain [(sentence, mention),...]

    total_amt = X.shape[0]
    del dataset, mentions # cleanup the memory
    
    # Parsing the labels and convert to integer using comma as separetor
    print("Creating MultiLabel..")
    temp = list()
    for element in y:
        values = element.split(',')
        values = list(map(int, values))
        temp.append(values)
    # Convert to np.array
    temp = np.array(temp)

    # Binarizer the labels
    print("Binarizering labels..")
    mlb = MultiLabelBinarizer(sparse_output=True)
    y = mlb.fit_transform(temp)
    print("MultiLable y shape:",y.shape)
    label_num = len(mlb.classes_)
    del temp
    print(" - Total number of labels: {:10d}".format(y.shape[1]))

    # Spliting training and testing data
    print("Spliting the dataset:")
    training = 1. - testing
    tr_amt = int(total_amt * training)
    te_amt = int(total_amt * testing)
    print("- Training Data: {:10d} ({:2.2f}%)".format(tr_amt, 100. * training))
    print("- Testing Data : {:10d} ({:2.2f}%)".format(te_amt, 100. * testing))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing, random_state=None)
    
    del X, y
    pkl.dump(y_train, open(model_dir + "training_label.pkl", 'wb'))
    pkl.dump(y_test, open(model_dir + "testing_label.pkl", 'wb'))
    del y_train, y_test

    X_train_mention = X_train[:, 1] # extract mention
    X_train = X_train[:, 0] # extract sentence
    X_test_mention = X_test[:, 1] # extract mention
    X_test = X_test[:, 0] # extract sentence
    
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

    pkl.dump(X_t, open(model_dir + "training_data.pkl", 'wb'))
    pkl.dump(X_te, open(model_dir + "testing_data.pkl", 'wb'))
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

    pkl.dump(X_m_t, open(model_dir + "training_mention.pkl", 'wb'))
    pkl.dump(X_m_te, open(model_dir + "testing_mention.pkl", 'wb'))
    del X_m_t, X_m_te


    print("dumping pickle file of tokenizer/m_tokenizer/mlb...")
    
    # dumping model
    pkl.dump(tokenizer, open(model_dir + "tokenizer.pkl", 'wb'))
    pkl.dump(m_tokenizer, open(model_dir + "m_tokenizer.pkl", 'wb'))
    pkl.dump(mlb, open(model_dir + "mlb.pkl", 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs='?', type=str, default="model/", 
                        help="Directory to store models. [Default: \"model/\"]")
    parser.add_argument("--input", help="Input dataset filename.")
    parser.add_argument("--test", nargs='?', const=0.1, type=float, default=0.1,
                        help="Specify the portion of the testing data to be split.\
                        [Default: 10\% of the entire dataset]")
    args = parser.parse_args()

    run(args.model, args.input, args.test)
