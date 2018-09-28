import pickle as pkl
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, dot, Permute, Reshape, multiply  # merge
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNLSTM
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPool2D, BatchNormalization
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from modules.fastText_model import fastText
import numpy as np


def Embedding_Layer(tokenizer,  # tokenizer_model,
                    max_num_words,
                    input_length,
                    embedding_dim,
                    filename=None,
                    preload=None):
    """
    Args:
        tokenizer_model(str): Pre-trained tokenizer for the data.
        max_num_words(int): Maximum number of words in a sentence.
        input_length(int): Input length of the model
        embedding_dim(int): The dimension of the embedding vectors.
        filename(str): Filename of pre-trained embeddings.
        preload(): Pre-loaded embedding object.
    Returns:
        embedding_layer(keras.layers.Embedding): Keras Embedding layer object.
        embeddings_index(): 
    """
    # Load trained tokenizer model
    # tokenizer = pkl.load(open(tokenizer_model, "rb"))
    word_index = tokenizer.word_index
    # Parameters for embedding layer
    num_words = min(max_num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    if filename is not None:
        # Parse embedding matrix
        if preload is not None:
            print("Pre-loaded embedding layer is given, use pre-loaded one.")
            embeddings_index = preload
        else:
            print("Loading pre-trained embedding model from {0}...".format(
                filename))
            embeddings_index = fastText(filename, dim=embedding_dim)

        print("Preparing embedding matrix...")
        # Process mention
        for word, idx in word_index.items():
            #
            if idx >= num_words:
                break
            # Fetch pre-trained vector
            embedding_vector = embeddings_index[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[idx] = embedding_vector
        # keras.layers.Embedding
        embedding_layer = Embedding(
            num_words,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=True)
    else:
        print(
            "No pre-trained embedding is given, training embedding from scratch."
        )
        embedding_layer = Embedding(
            num_words, embedding_dim, input_length=input_length)
        embeddings_index = None

    # Return embedding_layer only if reuse is not asserted
    return (embedding_layer, embeddings_index)


def attention_3d_block(inputs,
                       len_seq=40,
                       embedding_dim=100,
                       SINGLE_ATTENTION_VECTOR=False):
    """
    Args:
        inputs
        len_seq
        embedding_dim
        SINGLE_ATTENTION_VECTOR
    """
    a = Permute((2, 1))(inputs)
    # this line is not useful. It's just to know which dimension is what.
    a = Reshape((embedding_dim, len_seq))(a)
    a = Dense(len_seq, activation="softmax")(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name="dim_reduction")(a)
        a = RepeatVector(embedding_dim)(a)
    a_probs = Permute((2, 1), name="attention_vec")(a)

    output_attention_mul = multiply([inputs, a_probs], name="attention_mul")
    return output_attention_mul


def TextCNN(input,
            input_dim,
            embedding_dim,
            length,
            num_filters,
            filter_sizes,
            dropout,
            batch_norm=True):

    reshape_dim = input_dim + (1, )
    x = Reshape(reshape_dim)(input)

    def TextCNN_basic_block(input,
                            num_filters,
                            kernel_size,
                            pool_size,
                            dropout,
                            batch_norm=True):
        x = Conv2D(
            num_filters,
            kernel_size=kernel_size,
            padding="valid",
            kernel_initializer="normal",
            activation="relu")(input)
        if batch_norm:
            x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=pool_size, strides=(1, 1), padding="valid")(x)
        x = Dropout(dropout)(x)
        return x

    stream_0 = TextCNN_basic_block(
        input=x,
        num_filters=num_filters,
        kernel_size=(filter_sizes[0], embedding_dim),
        pool_size=(length - filter_sizes[0] + 1, 1),
        dropout=dropout,
        batch_norm=batch_norm)
    stream_1 = TextCNN_basic_block(
        input=x,
        num_filters=num_filters,
        kernel_size=(filter_sizes[1], embedding_dim),
        pool_size=(length - filter_sizes[1] + 1, 1),
        dropout=dropout,
        batch_norm=batch_norm)
    stream_2 = TextCNN_basic_block(
        input=x,
        num_filters=num_filters,
        kernel_size=(filter_sizes[2], embedding_dim),
        pool_size=(length - filter_sizes[2] + 1, 1),
        dropout=dropout,
        batch_norm=batch_norm)

    # Concatenate three streams and flatten
    output = concatenate([stream_0, stream_1, stream_2])
    output = Flatten()(output)
    return output


def BiLSTM(input):
    """
    Encapsulated Bi-Directional LSTM and GlobalMaxPool1D
    """
    x = Bidirectional(CuDNNLSTM(units=50, return_sequences=True))(input)
    x = GlobalMaxPool1D()(x)
    return x


def EntityTypingNet(architecture,
                    n_classes,
                    context_tokenizer,
                    mention_tokenizer,
                    desc_tokenizer,
                    context_emb=None,
                    context_embedding_dim=100,
                    mention_emb=None,
                    mention_embedding_dim=100,
                    desc_emb=None,
                    desc_embedding_dim=100,
                    same_emb=True,
                    n_words=100000,
                    n_mention=20000,
                    n_description=20000,
                    len_context=100,
                    len_mention=5,
                    len_description=100,
                    attention=False,
                    subword=False,
                    indicator=False,
                    description=False,
                    matching=False,
                    merge_mode="concatenate",
                    dropout=0.50,
                    use_softmax=False,
                    optimizer="adam",
                    learning_rate=0.001):
    """
    Arguments:
        architecture(str): Indicate which model to be used.
        n_classes(int): The number of class for the task.
        context_tokenizer(str): Path to context tokenizer.
        mention_tokenizer(str): Path to mention tokenizer.
        context_emb(str): Path to pretrained embedding for context.
        context_embedding_dim(int): Embedding dimension for context.
        mention_emb(str): Path to pretrained embedding for mention.
        mention_embedding_dim(int): Embedding dimension for mention.
        same_emb(bool): Use same given pretrained embedding or train
            both embedding from scrath. for both context and mention.
            Overrides same_dim when True. [Default: True]
        same_dim(bool): Use same embedding dimension for both context
            and mention. Ignore when same_emb is True. [Default: True]
        n_words
        n_mention
        len_context(int):
        len_mention(int):
        attention(bool): When asserted, attention layer would be added
            to the context stream.
        subword():
        indicator():
        merge_mode(str): The type of operation to combine features
            from context and mention.
        dropout(float): The dropout rate for the entire model.
        use_softmax(bool): When asserted, the final layer would be
            Softmax activation layer for single-class classification
            otherwise the activation layer would be Sigmoid and use
            0.5 as threshold to make predictions.
        optimizer(str): The type of optimizer to use.
        learning_rate(float): Learning rate for the optimizer.
    """
    #
    if same_emb:
        print("Using same embedding for both context and mention.")
        mention_emb = context_emb
        mention_embedding_dim = context_embedding_dim
    """
    TO-DOs: Make robust check on the given parameters
    if same_emb and (context_emb is None) and (mention_emb is None):

    if same_emb:
        if context_emb is not None and mention_emb is None:
            context_emb = mention_emb
        if context_emb != mention_emb:
            print("[Err] Given embedding parameters are ambiguous.")
            print(" * same_emb: {0}".format(same_emb))
            print(" - Context Embedding: {0}".format(context_emb))
            print(" - Mention Embedding: {0}".format(mention_emb))
            exit()
        if context_embedding_dim != mention_embedding_dim:
            print("[Err] Given embedding parameters are ambiguous.")
            print(" * same_emb: {0}".format(same_emb))
            print(" - Context embedding_dim: {0}".format(context_embedding_dim))
            print(" - Mention embedding_dim: {0}".format(mention_embedding_dim))
            exit()
    """
    #
    filter_sizes = [1, 2, 3]
    num_filters = 64
    architecture = architecture.lower()
    optimizer = optimizer.lower()
    # Two-stream inputs:
    # (1) Context (sentence)
    context = Input(shape=(len_context, ), name="Context")

    context_embedding, preload = Embedding_Layer(
        tokenizer=context_tokenizer,
        max_num_words=n_words,
        input_length=len_context,
        embedding_dim=context_embedding_dim,
        filename=context_emb)

    x_context = context_embedding(context)

    if attention:  # attention before lstm
        x_context = attention_3d_block(
            x_context,
            len_seq=len_context,
            embedding_dim=context_embedding_dim)
    else:
        pass

    # (2) Indicator or Mention
    if indicator:
        indicators = Input(shape=(len_context, ), name="Indicator")
        x_indicator = Reshape((len_context, 1))(indicators)
    else:
        # Embedding for mention/subword
        mention = Input(shape=(len_mention, ), name="Mention")
        # TO-DOs:
        # Vectorization on subwords

        mention_embedding, _ = Embedding_Layer(
            tokenizer=mention_tokenizer,
            max_num_words=n_mention,
            input_length=len_mention,
            embedding_dim=mention_embedding_dim,
            filename=mention_emb,
            preload=preload if same_emb else None)

        del preload

        x_mention = mention_embedding(mention)

    # (3) Description (if applicable as third input stream)
    if description:
        descrip = Input(shape=(len_description, ), name="Description")
        descrip_embedding, _ = Embedding_Layer(
            tokenizer=desc_tokenizer,
            max_num_words=n_description,
            input_length=len_description,
            embedding_dim=desc_embedding_dim,
            filename=None)

        x_description = descrip_embedding(descrip)

    # Bi-Directional LSTM
    if architecture == "blstm":
        if indicator:
            x_context = concatenate([x_context, x_indicator])
            x_context = BiLSTM(x_context)
        else:
            x_context = BiLSTM(x_context)
            x_mention = BiLSTM(x_mention)

        if description:
            x_description = BiLSTM(x_description)

    # Text CNN by Kim
    elif architecture == "cnn" or architecture == "text_cnn":
        # (1) Context
        if indicator:
            x_context = concatenate([x_context, x_indicator])
            x_context = TextCNN(
                input=x_context,
                input_dim=(len_context, context_embedding_dim + 1),
                embedding_dim=context_embedding_dim + 1,
                length=len_context,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                dropout=dropout,
                batch_norm=True)
        else:
            x_context = TextCNN(
                input=x_context,
                input_dim=(len_context, context_embedding_dim),
                embedding_dim=context_embedding_dim,
                length=len_context,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                dropout=dropout,
                batch_norm=True)
            x_mention = TextCNN(
                input=x_mention,
                input_dim=(len_mention, mention_embedding_dim),
                embedding_dim=mention_embedding_dim,
                length=len_mention,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                dropout=dropout,
                batch_norm=True)

        if description:
            x_description = TextCNN(
                input=x_description,
                input_dim=(len_description, desc_embedding_dim),
                embedding_dim=desc_embedding_dim,
                length=len_description,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                dropout=dropout,
                batch_norm=True)
    else:
        x_context = None
        x_mention = None

    # Concatenate
    if indicator:
        if description:
            x = concatenate([x_context, x_description])
            x = Dropout(dropout)(x)
        else:
            x = Dropout(dropout)(x_context)
    else:
        if merge_mode == "concatenate":
            x = concatenate([x_context, x_mention])
            x = Dropout(dropout)(x)
        # Inner product
        elif merge_mode == "dot":
            x = dot([x_context, x_mention], axes=-1)

    x = Dense(200, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    use_sigmoid = description or matching
    activation = "softmax" if (use_softmax and not use_sigmoid) else "sigmoid"
    # print(" - Last layer activation function: {}".format(activation))
    if description or matching:
        x = Dense(1, activation=activation)(x)
    else:
        x = Dense(n_classes, activation=activation)(x)

    if indicator:
        if description:
            model = Model(inputs=[context, indicators, descrip], outputs=x)
        else:
            model = Model(inputs=[context, indicators], outputs=x)
    else:
        model = Model(inputs=[context, mention], outputs=x)

    # Optimizer
    print("Using {0} optimizer (lr={1})".format(optimizer, learning_rate))
    opt = None
    if optimizer == "adam":
        opt = Adam(lr=learning_rate)
    elif optimizer == "rms":
        opt = RMSprop(lr=learning_rate)
    elif optimizer == "adagrad":
        opt = Adagrad(lr=learning_rate)
    elif optimizer == "sgd":
        opt = SGD(lr=learning_rate)

    if use_softmax:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])
    elif description:
        model.compile(
            loss="hinge",
            optimizer=opt,
            metrics=["accuracy"])
    else:
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])

    return model
