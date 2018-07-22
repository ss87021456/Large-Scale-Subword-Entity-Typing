import pandas as pd
import numpy as np
from scipy import sparse
import argparse
import pickle as pkl
from utils import split_data, create_embedding_layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support 
from fastText_model import fastText # pretrain-model
from keras.preprocessing import text, sequence
from keras import backend as K
from keras.layers import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from nn_model import BLSTM, CNN
from evaluation import just_test

# Training w/o pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --mode=[CNN,BLSTM]
# Training w/ pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --pre --emb=data/FastText_embedding.vec --mode=[CNN,BLSTM]

# Additional option --subword --attention
# /home/chiawei2/nlp_tool/fastText-0.1.0/vector/fastText_Pubmed.vec

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5 # 15 if subowrd else 5
EMBEDDING_DIM = 100

# Hyper-parameter
batch_size = 64
epochs = 5

# Set memory constraint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def run(model_dir, model_type, pre=False, embedding=None, subword=False, attention=False):
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Load models
    sb_tag = "w" if subword else "wo"
    mlb = pkl.load(open(model_dir + "mlb_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    tokenizer = pkl.load(open(model_dir + "tokenizer_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    m_tokenizer = pkl.load(open(model_dir + "m_tokenizer_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    
    word_index = tokenizer.word_index
    m_word_index = m_tokenizer.word_index
    label_num = len(mlb.classes_)

    ###
    tokenizer_model = model_dir + "tokenizer_{0}_subword_filter.pkl".format(sb_tag)
    m_tokenizer_model = model_dir + "m_tokenizer_{0}_subword_filter.pkl".format(sb_tag)
    ###
    embedding_layer, preload = create_embedding_layer(tokenizer_model=tokenizer_model,
                                             filename=embedding,
                                             max_num_words=MAX_NUM_WORDS,
                                             max_length=MAX_SEQUENCE_LENGTH,
                                             embedding_dim=EMBEDDING_DIM,
                                             reuse=True)
    m_embedding_layer = create_embedding_layer(tokenizer_model=m_tokenizer_model,
                                               filename=embedding,
                                               max_num_words=MAX_NUM_MENTION_WORDS,
                                               max_length=MAX_MENTION_LENGTH,
                                               embedding_dim=EMBEDDING_DIM,
                                               preload=preload)
    del preload

    # Building Model
    print("Building computational graph...")
    if model_type == "BLSTM":
        print("Building default BLSTM mode with attention:", attention, "subword:", subword)
        model = BLSTM(label_num=label_num,
                      sentence_emb=embedding_layer,
                      mention_emb=m_embedding_layer,
                      attention=attention,
                      subword=subword,
                      mode='concatenate',
                      dropout=0.1)
    elif model_type == "CNN":
        print("Building default CNN mode with attention:",attention,"subword:",subword)
        model = CNN(label_num=label_num,
                    sentence_emb=embedding_layer,
                    mention_emb=m_embedding_layer,
                    attention=attention,
                    subword=subword,
                    mode='concatenate',
                    dropout=0.1)

    print(model.summary())
    #exit()

    prefix = "{0}{1}".format("Subword-"   if subword   else "",
                             "Attention-" if attention else "")
    # for keras to save model each epoch
    file_path =  prefix + model_type + "-weights-{epoch:02d}.hdf5"
    # deal with model_name
    model_name = prefix + model_type + "-weights-00.hdf5"

    # Save every epoch
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early] #early

    # Training
    print("Loading validation data...")
    X_vali = pkl.load(open(model_dir + "validation_data_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    X_vali_mention = pkl.load(open(model_dir + "validation_mention_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    y_vali = pkl.load(open(model_dir + "validation_label_{0}_subword_filter.pkl".format(sb_tag), 'rb'))

    # Training
    print("Loading training data...")
    X_train = pkl.load(open(model_dir + "training_data_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    X_train_mention = pkl.load(open(model_dir + "training_mention_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    y_train = pkl.load(open(model_dir + "training_label_{0}_subword_filter.pkl".format(sb_tag), 'rb'))

    print("Begin training...")
    model.fit([X_train, X_train_mention],
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.01,
              callbacks=callbacks_list)

    # Evaluation
    record = 0
    index = 0
    print("Loading trained weights for validation...")
    with open("0721.txt", "a") as file_writer:
        for i in range(1, epochs + 1, 1):
            file = list(model_name)
            file[-6] = str(i)
            file = "".join(file)
            file_writer.write("{:s}\n".format(file))
            model.load_weights(file)
            print("Predicting...",file)
            y_pred = model.predict([X_vali, X_vali_mention])
        
            y_pred[y_pred >= 0.5] = 1.
            y_pred[y_pred < 0.5] = 0.
            y_pred = sparse.csr_matrix(y_pred)
        
            eval_types = ['micro','macro','weighted']
            for eval_type in eval_types:
                p, r, f, _ = precision_recall_fscore_support(y_vali, y_pred, average=eval_type)
                print("[{}]\t{:3.3f}\t{:3.3f}\t{:3.3f}".format(eval_type, p, r, f))
                file_writer.write("[{}]\t{:3.3f}\t{:3.3f}\t{:3.3f}\n".format(eval_type, p, r, f))
                if eval_type == 'micro' and record < f:
                    record = f
                    index = i

    # Test model with best micro F1 score
    file_path =  prefix + model_type + "-weights-{:02d}.hdf5".format(index)
    just_test(model=model, filename=file_path)

    K.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", action="store_true", help="Use pretrained embedding Model or not")
    parser.add_argument("--emb", help="please provide pretrained Embedding Model.")
    parser.add_argument("--subword", action="store_true" , help="Use subword or not")
    parser.add_argument("--attention",action="store_true", help="Use attention or not")
    parser.add_argument("--model", nargs='?', type=str, default="model/", 
                        help="Directory to load models. [Default: \"model/\"]")
    parser.add_argument("--mode", nargs='?', type=str, default="BLSTM",
                        help="different model architecture BLTSM or CNN [Default: \"BLSTM\"]")
    args = parser.parse_args()

    run(args.model, args.mode, args.pre, args.emb, args.subword, args.attention)
