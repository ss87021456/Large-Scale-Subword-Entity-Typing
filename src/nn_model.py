from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, dot
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout


def BLSTM(label_num, sentence_emb=None, mention_emb=None, mode='concatenate', dropout=0.1):
        
        MAX_NUM_WORDS = 30000
        MAX_NUM_MENTION_WORDS = 20000
        MAX_SEQUENCE_LENGTH = 40
        MAX_MENTION_LENGTH = 5
        EMBEDDING_DIM = 100

        sentence = Input(shape=(MAX_SEQUENCE_LENGTH, ), name='sentence')        
        
        # Pretrain sentence_embedding
        if sentence_emb is not None:
            x = sentence_emb(sentence)
        else:
            x = Embedding(MAX_NUM_WORDS,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH)(sentence)
        
        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)

        mention = Input(shape=(MAX_MENTION_LENGTH, ), name='mention')
        # Pretrain mention_embedding
        if mention_emb is not None:
            x_2 = mention_emb(mention)
        else:
            x_2 = Embedding(MAX_NUM_MENTION_WORDS,EMBEDDING_DIM,input_length=MAX_MENTION_LENGTH)(mention)
        
        x_2 = Bidirectional(LSTM(50, return_sequences=True))(x_2)
        x_2 = GlobalMaxPool1D()(x_2)

        if mode == 'concatenate':
            x = concatenate([x, x_2])           # Concatencate
            x = Dropout(dropout)(x)
        elif mode == 'dot':
            x = dot([x, x_2], axes=-1)           # Dot product

        x = Dense(200, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(label_num, activation="sigmoid")(x)
        model = Model(inputs=[sentence, mention], outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam')
        return model