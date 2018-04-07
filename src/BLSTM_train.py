import pandas as pd
import numpy as np
import argparse
import pickle as pkl
from utils import split_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from fastText_model import fastText # pretrain-model
from keras.preprocessing import text, sequence
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# CUDA_VISIBLE_DEVICES=6,7 python ./src/BLSTM_train.py --corpus=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --pre=False 
# CUDA_VISIBLE_DEVICES=6,7 python ./src/BLSTM_train.py --corpus=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --pre=True --emb=./data/model.vec
# CUDA_VISIBLE_DEVICES=6,7 python ./src/BLSTM_train.py --corpus=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --pre=True --emb=./data/model.vec --evaluation

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5
EMBEDDING_DIM = 100

# Hyper-parameter
batch_size = 64
epochs = 5

def e_precision(y_true, y_pred, mode="NORMAL"):
    if not mode in ["NORMAL", "INDEX"]:
        mode = "NORMAL"

    if mode == "NORMAL":
        comparison = np.equal(y_true, y_pred)
        return np.mean(np.sum(y_pred * comparison, axis=1) / np.sum(y_pred, axis=1))

    elif mode == "INDEX":
        prec_list = list()
        for itr in range(y_pred.shape[0]):
            indices,  = np.where(y_pred[itr, :] == 1)
            # Assign 0. precision if no label is asserted
            if indices.size == 0:
                prec_list.append(0.)
                continue
            else:
                prec_list.append(np.average(y_true[itr, indices]))
        #
        return np.average(prec_list)

def e_recall(y_true, y_pred, mode="NORMAL"):
    if not mode in ["NORMAL", "INDEX"]:
        mode = "NORMAL"

    if mode == "NORMAL":
        comparison = np.equal(y_true, y_pred)
        return np.mean(np.sum(y_true * comparison, axis=1) / np.sum(y_true, axis=1))

    elif mode == "INDEX":
        recall_list = list()
        for itr in range(y_true.shape[0]):
            indices,  = np.where(y_true[itr, :] == 1)
            # Assign 0. precision if no label is asserted
            if indices.size == 0:
                recall_list.append(0.)
                continue
            else:
                recall_list.append(np.average(y_pred[itr, indices]))
        #
        return np.average(recall_list)

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

def run(model_dir, filename, pre=True, embedding=None, testing=0.1, evaluation=False):
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Load models
    mlb = pkl.load(open(model_dir + "mlb.pkl", 'rb'))
    label_num = len(mlb.classes_)
    tokenizer = pkl.load(open(model_dir + "tokenizer.pkl", 'rb'))
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
        embedding_layer = Embedding(MAX_NUM_WORDS,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    )
    
    def measure_f1(y_true, y_pred):
        return
    def _precision(y_true, y_pred):
        comparison = K.cast(K.equal(y_true, y_pred), dtype='float32')
        return K.mean(K.sum(y_pred * comparison, axis=1) / K.sum(y_pred, axis=1))

    def _recall(y_true, y_pred):
        comparison = K.cast(K.equal(y_true, y_pred), dtype='float32')
        return K.mean(K.sum(y_true * comparison, axis=1) / K.sum(y_true, axis=1))

    def BLSTM():
        sentence = Input(shape=(MAX_SEQUENCE_LENGTH, ), name='sentence')        
        x = embedding_layer(sentence)
        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)

        mention = Input(shape=(MAX_MENTION_LENGTH, ), name='mention')
        x_2 = Embedding(MAX_NUM_MENTION_WORDS,
                            EMBEDDING_DIM,
                            input_length=MAX_MENTION_LENGTH)(mention)
        x_2 = Bidirectional(LSTM(50, return_sequences=True))(x_2)
        x_2 = GlobalMaxPool1D()(x_2)

        x = concatenate([x, x_2])
        x = Dropout(0.1)(x)
        x = Dense(200, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(label_num, activation="sigmoid")(x)
        model = Model(inputs=[sentence, mention], outputs=x)
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
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True, mode='min')
    
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    # early = EarlyStopping(mode="min", patience=20)
    
    callbacks_list = [checkpoint, early] #early

    if evaluation:
        print("Loading testing data...")
        X_test = pkl.load(open(model_dir + "testing_data.pkl", 'rb'))
        X_test_mention = pkl.load(open(model_dir + "testing_mention.pkl", 'rb'))
        y_test = pkl.load(open(model_dir + "testing_label.pkl", 'rb'))
        print("Loading trained weights for predicting...")
        model.load_weights(file_path)
    else:
        print("Loading training data...")
        X_train = pkl.load(open(model_dir + "training_data.pkl", 'rb'))
        X_train_mention = pkl.load(open(model_dir + "training_mention.pkl", 'rb'))
        y_train = pkl.load(open(model_dir + "training_label.pkl", 'rb'))
        print("Begin training...")
        model.fit([X_train, X_train_mention], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
        del X_train, y_train # clean up memory
        print("Loading testing data...")
        X_test = pkl.load(open(model_dir + "testing_data.pkl", 'rb'))
        y_test = pkl.load(open(model_dir + "testing_label.pkl", 'rb'))

    print("Predicting...")
    y_pred = model.predict([X_test, X_test_mention])
    del X_test
    # y_pred = model.predict(X_t)
    # print(list(y_pred[1, :]))
    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.
    # p = _precision(K.cast(y_test.toarray(), dtype='float32'), y_pred)
    slice_len = 100
    #times = len(y_pred)/slice_len
    print("Slicing...")
    Y_pred = split_data(y_pred, slice_len, mode="SINGLE")
    Y_test = split_data(y_test, slice_len, mode="SINGLE")
    del y_pred, y_test
    p_ = 0.
    r_ = 0.
    print("Calculate Precision Recall...")
    go = 0
    for (y_pred, y_test) in zip(Y_pred, Y_test):
        print("Processing {0} / {1} ".format(go, slice_len))
        go += 1
        # Implement it using pure numpy method instead of Tensor...
        p_ += e_precision(y_test.toarray(), y_pred, mode="INDEX")
        r_ += e_recall(y_test.toarray(), y_pred, mode="INDEX")
        del y_pred, y_test # clean up mem
        #p = _precision(K.cast(y_test.toarray(), dtype='float32'), y_pred)
        #r = _recall(K.cast(y_test.toarray(), dtype='float32'), y_pred)
    print("Precision: {0} | Recall: {1}".format(100. * (p_/slice_len), 100. * (r_/slice_len)))
    # print(list(y_test.toarray()[0, :]))
    # print(list(y_pred[0, :]))
    print(np.sum(y_test.toarray()))
    print(np.sum(y_pred))
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
    parser.add_argument("--evaluation", action="store_true", help="Evaluation mode.")
    parser.add_argument("--test", nargs='?', const=0.1, type=float, default=0.1,
                        help="Specify the portion of the testing data to be split.\
                        [Default: 10\% of the entire dataset]")
    parser.add_argument("--model", nargs='?', type=str, default="model/", 
                        help="Directory to load models. [Default: \"model/\"]")
    args = parser.parse_args()

    run(args.model, args.corpus, args.pre, args.emb, args.test, args.evaluation)
