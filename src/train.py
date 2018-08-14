from scipy import sparse
import argparse
import numpy as np
import pickle as pkl
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from evaluation import just_test, predict
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from modules.entity_net import EntityTypingNet

# Training w/o pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --arch=[CNN,BLSTM]
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --arch=[CNN,BLSTM] --data_tag=kbp
# Training w/ pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=data/FastText_embedding.vec --arch=[CNN,BLSTM]
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=data/FastText_embedding.vec --arch=Text_CNN
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=/shared/data/embed/w2v.txt --arch=[CNN,BLSTM] --data_tag=kbp

# Additional option --subword --attention
# /home/chiawei2/nlp_tool/fastText-0.1.0/vector/fastText_Pubmed.vec

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5  # 15 if subowrd else 5

# Set memory constraint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def run(args):
    postfix = ("_" + args.data_tag) if args.data_tag is not None else ""
    args.tag = ("_" + args.tag) if args.tag is not None else ""

    # Parse directory name
    if not args.model_dir.endswith("/"):
        args.model_dir += "/"

    #########################################
    # Load models (TO-BE-REVISED)
    sb_tag = "w" if args.subword else "wo"
    mlb_ = "mlb_{0}_subword_filter{1}.pkl".format(sb_tag, postfix)
    mlb = pkl.load(open(args.model_dir + mlb_, 'rb'))
    n_classes = len(mlb.classes_)
    args.context_tokenizer = args.model_dir + "tokenizer_{0}_subword_filter{1}.pkl".format(
        sb_tag, postfix)
    args.mention_tokenizer = args.model_dir + "m_tokenizer_{0}_subword_filter{1}.pkl".format(
        sb_tag, postfix)
    #########################################
    # print(args)

    # Building Model
    print("Building computational graph...")

    model = EntityTypingNet(
        architecture=args.arch,
        n_classes=n_classes,
        context_tokenizer=args.context_tokenizer,
        mention_tokenizer=args.mention_tokenizer,
        context_emb=args.context_emb,
        context_embedding_dim=args.context_embedding_dim,
        mention_emb=args.mention_emb,
        mention_embedding_dim=args.mention_embedding_dim,
        same_emb=args.same_emb,
        n_words=MAX_NUM_WORDS,
        n_mention=MAX_NUM_MENTION_WORDS,
        len_context=MAX_SEQUENCE_LENGTH,
        len_mention=MAX_MENTION_LENGTH,
        attention=args.attention,
        merge_mode=args.merge_mode,
        dropout=args.dropout,
        subword=args.subword,
        use_softmax=args.use_softmax,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate)

    print(model.summary())

    prefix = "{0}{1}".format("Subword-" if args.subword else "", "Attention-"
                             if args.attention else "")
    # Save weights at each epoch
    file_path = prefix + args.arch + "-weights-{epoch:02d}" + "{:s}.hdf5".format(
        args.tag)

    # Save every epoch
    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early]

    # Training
    print("Loading training data...")
    X_train = pkl.load(
        open(
            args.model_dir + "training_data_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    X_m_train = pkl.load(
        open(
            args.model_dir +
            "training_mention_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    y_train = pkl.load(
        open(
            args.model_dir + "training_label_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))

    #if category:
    #    y_train =  np.array(mlb.inverse_transform(y_train)).flatten()

    print("Begin training...")
    model.fit(
        [X_train, X_m_train],
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.01,
        callbacks=callbacks_list)

    # Evaluation
    record = 0
    index = 0
    # Validation data
    print("\nLoading validation data...")
    X_val = pkl.load(
        open(
            args.model_dir +
            "validation_data_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    X_m_val = pkl.load(
        open(
            args.model_dir +
            "validation_mention_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    y_val = pkl.load(
        open(
            args.model_dir +
            "validation_label_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))

    print("Loading trained weights for validation...")
    for i in range(1, args.epochs + 1, 1):
        # Deal with model_name for each epoch
        model_name = prefix + args.arch + "-weights-{:02d}{:s}.hdf5".format(
            i, args.tag)
        model.load_weights(model_name)

        f = predict(
            model,
            X_val,
            X_m_val,
            y_val,
            model_name,
            "results.txt",
            return_mf1=True,
            use_softmax=args.use_softmax)

        # Always choose model trained with more epoch when the F-1 score is same
        if record <= f:
            record = f
            index = i

    print("\nValidation completed, best micro-F1 score is at epoch {:02d}".
          format(index))
    # Test model with best micro F1 score
    model_name = prefix + args.arch + "-weights-{:02d}{:s}.hdf5".format(
        index, args.tag)
    just_test(
        model=model,
        filename=model_name,
        subword=args.subword,
        postfix=postfix,
        use_softmax=args.use_softmax)

    K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Tokenizers
    parser.add_argument(
        "--context_tokenizer", type=str, help="Path to context_tokenizer.")
    parser.add_argument(
        "--mention_tokenizer", type=str, help="Path to mention_tokenizer.")
    # Embedding configurations
    parser.add_argument(
        "--same_emb",
        action="store_true",
        help="Use same configuration on embedding for both stream.")
    parser.add_argument(
        "--context_emb", help="Pretrained embedding model for context.")
    parser.add_argument(
        "--context_embedding_dim",
        type=int,
        default=100,
        help="Embedding dimension for context embedding layer.")
    parser.add_argument(
        "--mention_emb", help="Pretrained embedding model for mention.")
    parser.add_argument(
        "--mention_embedding_dim",
        type=int,
        default=100,
        help="Embedding dimension for mention embedding layer.")
    # Model hyperparameters
    parser.add_argument(
        "--arch",
        type=str,
        default="BLSTM",
        help="Different model architecture BLTSM or CNN [Default: \"BLSTM\"]")
    parser.add_argument(
        "--attention", action="store_true", help="Use attention or not")
    parser.add_argument(
        "--subword", action="store_true", help="Use subword or not")
    parser.add_argument(
        "--merge_mode",
        type=str,
        default="concatenate",
        help="Method to combine features from two-stream.")
    parser.add_argument(
        "--use_softmax",
        action="store_true",
        help="Perform single-class classification.")
    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=64, help="MiniBatch size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Choose optimizer \"adam\", \"RMS\", \"Adagrad\", \"SGD\".")
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate for the model.")
    # Others
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model/",
        help="Directory to load models. [Default: \"model/\"]")
    parser.add_argument(
        "--tag", type=str, help="Extra name tag on the saved model.")
    parser.add_argument(
        "--data_tag", type=str, help="Extra name tag on the dataset.")
    args = parser.parse_args()

    run(args)
