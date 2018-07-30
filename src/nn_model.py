from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, dot, Permute, Reshape, merge
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNLSTM
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPool2D


def attention_3d_block(inputs, max_seq_len=40, embedding_dim=100, SINGLE_ATTENTION_VECTOR=False):
    # SINGLE_ATTENTION_VECTOR = False
    # MAX_SEQUENCE_LENGTH = 40
    # EMBEDDING_DIM = 100
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((embedding_dim, max_seq_len))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(max_seq_len, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(embedding_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def BLSTM(label_num, embedding_dim=100, n_words=30000, n_mention=20000, len_seq=40, len_mention=5,
          sentence_emb=None, mention_emb=None, attention=False, mode='concatenate', dropout=0.1, subword=False):
    
    # MAX_NUM_WORDS = 30000
    # MAX_NUM_MENTION_WORDS = 11626#20000
    # MAX_SEQUENCE_LENGTH = 40
    # MAX_MENTION_LENGTH = 15 if subword else 5
    # EMBEDDING_DIM = 100
    sentence = Input(shape=(len_seq, ), name='sentence')
    
    # Pretrain sentence_embedding
    if sentence_emb is not None:
        x = sentence_emb(sentence)
    else:
        x = Embedding(n_words,embedding_dim, input_length=len_seq)(sentence)

    if attention: # attention before lstm
        x = attention_3d_block(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)

    # Embedding for mention/subword
    mention = Input(shape=(len_mention, ), name='mention')
    # Vectorization on subwords
    # Pretrain mention_embedding

    if mention_emb is not None:
        x_2 = mention_emb(mention)
    else:
        x_2 = Embedding(n_mention, embedding_dim,
                        input_length=len_mention)(mention)
    
    x_2 = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x_2)
    x_2 = GlobalMaxPool1D()(x_2)

    # Concatenate
    if mode == 'concatenate':
        x = concatenate([x, x_2])
        x = Dropout(dropout)(x)
    # Inner product
    elif mode == 'dot':
        x = dot([x, x_2], axes=-1)

    x = Dense(200, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(label_num, activation="sigmoid")(x)
    model = Model(inputs=[sentence, mention], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def CNN(label_num, embedding_dim=100, n_words=30000, n_mention=20000, len_seq=40, len_mention=5,
        sentence_emb=None, mention_emb=None, attention=False, mode='concatenate', dropout=0.1, subword=False):
    
    # n_words = 30000
    # MAX_NUM_MENTION_WORDS = 11626#20000
    # MAX_SEQUENCE_LENGTH = 40
    # MAX_MENTION_LENGTH = 15 if subword else 5
    # EMBEDDING_DIM = 100
    num_filters = 64

    sentence = Input(shape=(len_seq, ), name='sentence')        
    
    # Pretrain sentence_embedding
    if sentence_emb is not None:
        x = sentence_emb(sentence)
    else:
        x = Embedding(n_words,embedding_dim,input_length=len_seq)(sentence)
    
    if attention: # attention before lstm
        x = attention_3d_block(x)

    x = Conv1D(num_filters, 5, activation='relu', padding='valid')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(num_filters, 5, activation='relu', padding='valid')(x)
    x = GlobalMaxPool1D()(x)

    mention = Input(shape=(len_mention, ), name='mention')
    # Pretrain mention_embedding
    if mention_emb is not None:
        x_2 = mention_emb(mention)
    else:
        x_2 = Embedding(n_mention,embedding_dim,input_length=len_mention)(mention)

    x_2 = Conv1D(num_filters, 5, activation='relu', padding='same')(x_2)
    x_2 = MaxPooling1D(2)(x_2)
    x_2 = Conv1D(num_filters, 5, activation='relu', padding='same')(x_2)
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
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def Text_CNN(label_num, embedding_dim=100, n_words=30000, n_mention=20000, len_seq=40, len_mention=5,
             sentence_emb=None, mention_emb=None, attention=False, mode='concatenate', dropout=0.1, subword=False):
    
    # MAX_NUM_WORDS = 30000
    # MAX_NUM_MENTION_WORDS = 20000
    # MAX_SEQUENCE_LENGTH = 40
    # MAX_MENTION_LENGTH = 15 if subword else 5
    # EMBEDDING_DIM = 100
    # Text_CNN Configuration
    filter_sizes = [1,2,3]
    num_filters = 64

    sentence = Input(shape=(len_seq, ), name='sentence')        
    
    # Pretrain sentence_embedding
    if sentence_emb is not None:
        x = sentence_emb(sentence)
    else:
        x = Embedding(n_words,embedding_dim,input_length=len_seq)(sentence)
    
    if attention: # attention before lstm
        x = attention_3d_block(x)

    reshape = Reshape((len_seq,embedding_dim,1))(x)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    maxpool_0 = MaxPool2D(pool_size=(len_seq - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(len_seq - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(len_seq - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
    maxpool_0 = Dropout(dropout)(maxpool_0)
    maxpool_1 = Dropout(dropout)(maxpool_1)
    maxpool_2 = Dropout(dropout)(maxpool_2)
    content_vec = concatenate([maxpool_0, maxpool_1, maxpool_2])
    content_vec = Flatten()(content_vec)


    mention = Input(shape=(len_mention, ), name='mention')
    # Pretrain mention_embedding
    if mention_emb is not None:
        x_2 = mention_emb(mention)
    else:
        x_2 = Embedding(n_mention,embedding_dim,input_length=len_mention)(mention)
    
    reshape_2 = Reshape((len_mention,embedding_dim,1))(x_2)
    conv_0_2 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_2)
    conv_1_2 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_2)
    conv_2_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_2)
    maxpool_0_2 = MaxPool2D(pool_size=(len_mention - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0_2)
    maxpool_1_2 = MaxPool2D(pool_size=(len_mention - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1_2)
    maxpool_2_2 = MaxPool2D(pool_size=(len_mention - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2_2)
    maxpool_0_2 = Dropout(dropout)(maxpool_0_2)
    maxpool_1_2 = Dropout(dropout)(maxpool_1_2)
    maxpool_2_2 = Dropout(dropout)(maxpool_2_2)
    content_vec_2 = concatenate([maxpool_0_2, maxpool_1_2, maxpool_2_2])
    content_vec_2 = Flatten()(content_vec_2)

    if mode == 'concatenate':
        x = concatenate([content_vec, content_vec_2])           # Concatencate
        x = Dropout(dropout)(x)
    elif mode == 'dot':
        x = dot([x, x_2], axes=-1)           # Dot product

    x = Dense(200, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(label_num, activation="sigmoid")(x)
    model = Model(inputs=[sentence, mention], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
