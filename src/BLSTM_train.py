import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from fastText_model import fastText # pretrain-model
from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


# CUDA_VISIBLE_DEVICES=6,7 python ./src/BLSTM_train.py --corpus=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --pre=True --emb=./data/model.vec


# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 100

# Hyper-parameter
batch_size = 64
epochs = 2


def run(filename, pre=True, embedding=None):
    # filename = '../input/smaller_preprocessed_sentence_keywords_labeled.tsv'
    print("reading training dataset..")
    dataset = pd.read_csv(filename, sep='\t', names=['label','context'])
    list_sentences_train = dataset['context'].values[:100000]
    y_train = dataset['label'].values[:100000]
    del dataset # cleanup the memory
    
    print("Creating MultiLabel..")
    temp = list()
    for element in y_train:
        values = element.split(',')
        values = list(map(int, values))
        temp.append(values)
    
    temp = np.array(temp)
    
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(temp)
    print(y_train.shape, y_train[:5])
    label_num = len(mlb.classes_)
    del temp
    
    print("Tokenize sentences...")
    tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
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
                      metrics=['accuracy'])
        return model
    
    model = BLSTM()
    
    file_path="weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    
    callbacks_list = [checkpoint, early] #early
    print("Start training...")
    model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
    
    #model.load_weights(file_path)
    
    #y_test = model.predict(X_te)
    
    ''' Evaluation
    y_test = model.predict(X_te)
    y_test[y_test>=0.5] = 1
    y_test[y_test<0.5] = 0
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Input training data.")
    parser.add_argument("--pre", type=bool, help="Use pretrained embedding Model or not")
    parser.add_argument("--emb", help="please provide pretrained Embedding Model.")
    args = parser.parse_args()

    run(args.corpus, args.pre, args.emb)
