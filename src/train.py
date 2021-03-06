from scipy import sparse
import argparse
import numpy as np
import pickle as pkl
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from evaluation import just_test, predict
from utils import load_pkl_data
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from modules.entity_net import EntityTypingNet
"""
Training w/o pretrained
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --arch=[CNN,BLSTM]
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --arch=[CNN,BLSTM] --data_tag=kbp

Training w/ pretrained
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=data/FastText_embedding.vec --arch=[CNN,BLSTM]
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=data/FastText_embedding.vec --arch=Text_CNN
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --emb=/shared/data/embed/w2v.txt --arch=[CNN,BLSTM] --data_tag=kbp

Additional option --subword --attention
/home/chiawei2/nlp_tool/fastText-0.1.0/vector/fastText_Pubmed.vec

Train with description
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --arch=cnn --tokenizers=model/tokenizers_kbp.pkl --data_tag=kbp

"""

# Feature parameters
MAX_NUM_WORDS = 100000
MAX_NUM_MENTION_WORDS = 20000
MAX_NUM_DESCRIPTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
MAX_MENTION_LENGTH = 5  # 15 if subword else 5
MAX_DESCRIPTION_LENGTH = 100

# Set memory constraint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def run(args):
    # Add underscore to the tag
    args.tag = ("_" + args.tag) if args.tag is not None else ""
    # Parse prefix and postfix
    prefix = "{}{}".format("-Subword" if args.subword else "", "-Attention"
                             if args.attention else "")

    postfix = "{}{}{}".format("_subword" if args.subword else "",
                             ("_" + args.data_tag) if args.data_tag is not None else "",
                             ("_d" if args.description else ""))

    # Parse directory name
    if not args.model_dir.endswith("/"):
        args.model_dir += "/"
    if args.matching:
        print("Matching problem.")
    #########################################
    # Load models (TO-BE-REVISED)
    tokenizers = pkl.load(open(args.tokenizers, "rb"))
    n_classes = len(tokenizers["mlb"].classes_)
    try:
        desc_tokenizer = tokenizers["description"]
    except:
        desc_tokenizer = None
    #########################################
    # Building Model
    print("Building computational graph...")

    model = EntityTypingNet(
        architecture=args.arch,
        n_classes=n_classes,
        context_tokenizer=tokenizers["context"],
        mention_tokenizer=tokenizers["mention"],
        desc_tokenizer=desc_tokenizer,
        context_emb=args.context_emb,
        context_embedding_dim=args.context_embedding_dim,
        mention_emb=args.mention_emb,
        mention_embedding_dim=args.mention_embedding_dim,
        desc_emb=args.desc_emb,
        desc_embedding_dim=args.desc_embedding_dim,
        same_emb=args.same_emb,
        n_words=MAX_NUM_WORDS,
        n_mention=MAX_NUM_MENTION_WORDS,
        n_description=MAX_NUM_DESCRIPTION_WORDS,
        len_context=MAX_SEQUENCE_LENGTH,
        len_mention=MAX_MENTION_LENGTH,
        len_description=MAX_DESCRIPTION_LENGTH,
        attention=args.attention,
        subword=args.subword,
        indicator=args.indicator,
        description=False, # args.description,
        matching=args.matching,
        merge_mode=args.merge_mode,
        dropout=args.dropout,
        use_softmax=args.use_softmax,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate)

    print(model.summary())

    # Save weights at the end of each epoch
    save_prefix = "{:s}{:s}-weights{:s}".format(args.arch, prefix, args.tag)
    filename = save_prefix + "-{epoch:02d}.hdf5"

    checkpoint = ModelCheckpoint(
        filename,
        monitor="val_loss",
        verbose=1,
        save_best_only=False,
        mode="min")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early]

    X_train, Z_train, y_train, D_train = load_pkl_data(
        args.model_dir, "training", postfix, indicator=args.indicator, matching=args.matching)
    ######################################################
    """
    print(X_train.shape, y_train.shape)
    print("Stacking positive samples")
    n_instance = X_train.shape[0] // 6
    idxs = [i * 6 for i in range(n_instance)]
    tmp = np.vstack([X_train[idxs] for _ in range(4)])
    X_train = np.vstack([X_train, tmp])
    del tmp
    tmp = np.vstack([Z_train[idxs] for _ in range(4)])
    Z_train = np.vstack([Z_train, tmp])
    del tmp
    tmp = np.hstack([y_train[idxs] for _ in range(4)])
    y_train = np.hstack([y_train, tmp])
    del tmp
    if args.description:
        tmp = np.vstack([D_train[idxs] for _ in range(4)])
        D_train = np.vstack([D_train, tmp])
    """
    ######################################################
    # input = [X_train, Z_train]
    print(X_train.shape, Z_train.shape, y_train.shape)

    #if args.use_softmax:
    #    y_train =  np.array(mlb.inverse_transform(y_train)).flatten()
    input = [X_train, Z_train, D_train] if args.description else [X_train, Z_train]
    print("Begin training...")
    model.fit(
        input,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.01,
        callbacks=callbacks_list)

    # Evaluation
    record = 0
    index = 0

    X_val, Z_val, y_val, D_val = load_pkl_data(
        args.model_dir, "validation", postfix, indicator=args.indicator, description=args.description)

    print("Loading trained weights for validation...")
    for i in range(1, args.epochs + 1, 1):
        # Deal with model_name for each epoch
        model_name = "{:s}-{:02d}.hdf5".format(save_prefix, i)
        model.load_weights(model_name)

        f = predict(
            model,
            X_val,
            Z_val,
            y_val,
            model_name,
            "results.txt",
            return_mf1=True,
            use_softmax=args.use_softmax)

        # Always choose model trained with more epoch when the F-1 score is same
        if record <= f:
            record = f
            index = i

    print("\n * Best micro-F1 at Validation: epoch #{:02d}".format(index))
    # Test model with best micro F1 score
    model_name = "{:s}-{:02d}.hdf5".format(save_prefix, index)
    just_test(
        model=model,
        filename=model_name,
        postfix=postfix,
        use_softmax=args.use_softmax,
        indicator=args.indicator)

    K.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Tokenizers
    parser.add_argument(
        "--tokenizers", type=str, help="Path to tokenizers.")

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
    parser.add_argument(
        "--desc_emb", help="Pretrained embedding model for mention.")
    parser.add_argument(
        "--desc_embedding_dim",
        type=int,
        default=100,
        help="Embedding dimension for description embedding layer.")
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
        "--indicator", action="store_true", help="Use indicator or not")
    parser.add_argument(
        "--matching",
        action="store_true",
        help="Matching problem")
    parser.add_argument(
        "--merge_mode",
        type=str,
        default="concatenate",
        help="Method to combine features from two-stream.")
    parser.add_argument(
        "--use_softmax",
        action="store_true",
        help="Perform single-class classification.")
    parser.add_argument(
        "--description",
        action="store_true",
        help="Using zero-shot learning method.")
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
