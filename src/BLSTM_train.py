import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from fastText_model import fastText # pretrain-model
from keras.preprocessing import text, sequence
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# CUDA_VISIBLE_DEVICES=6,7 python ./src/BLSTM_train.py --corpus=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --pre=True --emb=./data/model.vec


# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 100

# Hyper-parameter
batch_size = 64
epochs = 2

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.f1s = []
        self.recalls = []
        self.precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _f1 = f1_score(val_targ, val_predict)
        _recall = recall_score(val_targ, val_predict)
        _precision = precision_score(val_targ, val_predict)
        self.f1s.append(_f1)
        self.recalls.append(_recall)
        self.precisions.append(_precision)
        print(" — F1: {:f} — Precision: {:f} — Recall {:f}".format(_f1, _precision, _recall))
        return

def run(filename, pre=True, embedding=None, testing=0.1):
    # filename = '../input/smaller_preprocessed_sentence_keywords_labeled.tsv'
    print("Loading dataset..")
    dataset = pd.read_csv(filename, sep='\t', names=['label','context'])

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
    print(y.shape, y[:5])
    label_num = len(mlb.classes_)
    del temp

    # Spliting training and testing data
    print("Spliting the dataset:")
    training = 1. - testing
    print("- Training Data: {:10d} ({:2.2f}%)".format(int(total_amt * training), 100. * training))
    print("- Testing Data : {:10d} ({:2.2f}%)".format(int(total_amt * testing), 100. * testing))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing, random_state=None)

    print("Tokenize sentences...")
    tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(list(X_train))
    list_tokenized_train = tokenizer.texts_to_sequences(X_train)
    # Padding sentences
    print("Padding sentences...")
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)
    
    word_index = tokenizer.word_index
    if pre:
        print("Loading pre-trained embedding model...")
        embeddings_index = fastText(embedding)

        print("Preparing embedding matrix...")
        # prepare embedding matrix
        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)
    else:
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    )
    
    def measure_f1(y_true, y_pred):
        return
    def _precision(y_true, y_pred):
        comparison = K.cast(K.equal(y_true, y_pred), dtype='float32')
        return K.sum(comparison) # K.sum(y_pred * comparison) / K.sum(y_pred)

    def _recall(y_true, y_pred):
        comparison = K.cast(K.equal(y_true, y_pred), dtype='float32')
        return K.sum(y_true * comparison) / K.sum(y_true)

    def BLSTM():
        inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
        x = embedding_layer(inp)
        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(label_num, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[_precision, _recall])
                      # metrics=['accuracy'])
        return model
    
    # Building Model
    print("Building computational graph...")
    model = BLSTM()
    # metrics = Metrics()
    print(model.summary())
    
    file_path="weights_base.best.hdf5"
    # checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True, mode='min')
    
    # early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    early = EarlyStopping(mode="min", patience=20)
    
    callbacks_list = [checkpoint, early] #early
    print("Begin training...")
    model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
    
    #model.load_weights(file_path)
    
    #y_test = model.predict(X_te)
    
    ''' Evaluation
    y_test = model.predict(X_te)
    y_test[y_test>=0.5] = 1
    y_test[y_test<0.5] = 0
    '''
    K.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Input training data.")
    parser.add_argument("--pre", type=bool, help="Use pretrained embedding Model or not")
    parser.add_argument("--emb", help="please provide pretrained Embedding Model.")
    parser.add_argument("--test", nargs='?', const=0.1, type=float, default=0.1,
                        help="Specify the portion of the testing data to be split.\
                        [Default: 10\% of the entire dataset]")
    args = parser.parse_args()
    run(args.corpus, args.pre, args.emb, args.test)
