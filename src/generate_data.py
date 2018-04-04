import pandas as pd
import numpy as np
import argparse
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing import text, sequence

# python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 100

def run(input, testing):
    print("Loading dataset..")
    dataset = pd.read_csv(input, sep='\t', names=['label','context'])
    
    X = dataset['context'].values[:] 
    y = dataset['label'].values[:]
    total_amt = X.shape[0]
    del dataset # cleanup the memory
    
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
    # print(y.shape, y[:5])
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
    
    #X_train = X[:tr_amt]
    #X_test = X[-tr_amt:]
    
    print("Tokenize sentences...")
    tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(list(X_train))
    list_tokenized_train = tokenizer.texts_to_sequences(X_train)
    list_tokenized_test = tokenizer.texts_to_sequences(X_test)
    # Padding sentences
    print("Padding sentences...")
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_SEQUENCE_LENGTH)
    print("dumping pickle file of [train/test] X, y, and tokenizer...")
    pkl.dump(X_t, open("training_data.pkl", 'wb'))
    pkl.dump(X_te, open("testing_data.pkl", 'wb'))
    pkl.dump(y_train, open("training_label.pkl", 'wb'))
    pkl.dump(y_test, open("testing_label.pkl", 'wb'))
    pkl.dump(tokenizer, open("tokenizer.pkl", 'wb'))
    pkl.dump(mlb, open("mlb.pkl", 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input dataset filename.")
    parser.add_argument("--test", nargs='?', const=0.1, type=float, default=0.1,
                        help="Specify the portion of the testing data to be split.\
                        [Default: 10\% of the entire dataset]")
    args = parser.parse_args()
    run(args.input, args.test)
