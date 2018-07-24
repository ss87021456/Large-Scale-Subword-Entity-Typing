from scipy import sparse
import argparse
import pickle as pkl
from utils import create_embedding_layer
from sklearn.metrics import precision_recall_fscore_support 
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nn_model import BLSTM, CNN, Text_CNN
from evaluation import just_test, predict

# Training w/o pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --arch=[CNN,BLSTM]
# Training w/ pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=data/FastText_embedding.vec --arch=[CNN,BLSTM]
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=data/FastText_embedding.vec --arch=Text_CNN

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

def run(model_dir, model_type, embedding=None, subword=False, attention=False, tag=None):
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

    m_embedding_layer, _ = create_embedding_layer(tokenizer_model=m_tokenizer_model,
                                               filename=embedding,
                                               max_num_words=MAX_NUM_MENTION_WORDS,
                                               max_length=MAX_MENTION_LENGTH,
                                               embedding_dim=EMBEDDING_DIM,
                                               preload=preload)
    del preload
    # exit()
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
    elif model_type == "Text_CNN":
        print("Building default Text_CNN mode with attention:",attention,"subword:",subword)
        model = Text_CNN(label_num=label_num,
                    sentence_emb=embedding_layer,
                    mention_emb=m_embedding_layer,
                    attention=attention,
                    subword=subword,
                    mode='concatenate',
                    dropout=0.5)

    print(model.summary())

    prefix = "{0}{1}".format("Subword-"   if subword   else "",
                             "Attention-" if attention else "")
    # for keras to save model each epoch
    file_path =  prefix + model_type + "-weights-{epoch:02d}" + "{:s}.hdf5".format(tag)

    # Save every epoch
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early] #early

    # Training
    print("Loading training data...")
    X_train = pkl.load(open(model_dir + "training_data_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    X_m_train = pkl.load(open(model_dir + "training_mention_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    y_train = pkl.load(open(model_dir + "training_label_{0}_subword_filter.pkl".format(sb_tag), 'rb'))

    print("Begin training...")
    model.fit([X_train, X_m_train],
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.01,
              callbacks=callbacks_list)

    # Evaluation
    record = 0
    index = 0
    # Validation data
    print("\nLoading validation data...")
    X_val = pkl.load(open(model_dir + "validation_data_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    X_m_val = pkl.load(open(model_dir + "validation_mention_{0}_subword_filter.pkl".format(sb_tag), 'rb'))
    y_val = pkl.load(open(model_dir + "validation_label_{0}_subword_filter.pkl".format(sb_tag), 'rb'))

    print("Loading trained weights for validation...")
    for i in range(1, epochs + 1, 1):
        # Deal with model_name for each epoch
        model_name = prefix + model_type + "-weights-{:02d}{:s}.hdf5".format(i, tag)
        model.load_weights(model_name)

        f = predict(model, X_val, X_m_val, y_val, model_name, "results.txt", return_mf1=True)

        if record < f:
            record = f
            index = i

    print("\nValidation completed, best micro-F1 score is at epoch {:02d}".format(index))
    # Test model with best micro F1 score
    model_name =  prefix + model_type + "-weights-{:02d}{:s}.hdf5".format(index, tag)
    just_test(model=model, filename=model_name, subword=subword)

    K.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", help="please provide pretrained Embedding Model.")
    parser.add_argument("--subword", action="store_true" , help="Use subword or not")
    parser.add_argument("--attention",action="store_true", help="Use attention or not")
    parser.add_argument("--model", nargs='?', type=str, default="model/", 
                        help="Directory to load models. [Default: \"model/\"]")
    parser.add_argument("--arch", nargs='?', type=str, default="BLSTM",
                        help="Different model architecture BLTSM or CNN [Default: \"BLSTM\"]")
    parser.add_argument("--tag", nargs='?', type=str, help="Extra name tag on the saved model.")
    args = parser.parse_args()

    run(args.model, args.arch, args.emb, args.subword, args.attention, args.tag)
