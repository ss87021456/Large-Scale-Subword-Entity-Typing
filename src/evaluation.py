import pandas as pd
import numpy as np
from scipy import sparse
import argparse, json
import pickle as pkl
from utils import split_data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from keras import backend as K
from keras.layers import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from nn_model import BLSTM, CNN
from tqdm import tqdm
from time import time
from keras.metrics import categorical_accuracy
from datetime import datetime

# testing
# CUDA_VISIBLE_DEVICES=1 python ./src/evaluation.py  --model_path=... --model_type=[BLSTM,CNN] [--subword] [--attention]

# visualize
# CUDA_VISIBLE_DEVICES=1 python ./src/evaluation.py --model_path=... --model_type=[BLSTM,CNN] [--subword] [--attention] --visualize > visualize.txt

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 11626  # 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5  # 15 if subowrd else 5
EMBEDDING_DIM = 100

# Hyper-parameter
batch_size = 64
epochs = 5


def p_r_f(pred, label):
    overlap_count = len(set(pred) & set(label))
    if len(pred) != 0:
        precision = overlap_count / len(pred)
    else:
        precision = 0

    recall = overlap_count / len(label)
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.

    return precision, recall, f1


# Set memory constraint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def run(model_dir,
        model_type,
        model_path,
        subword=False,
        attention=False,
        data_tag=None,
        visualize=False,
        pre=False):
    postfix = ("_" + data_tag) if data_tag is not None else ""
    print(model_dir, model_type, model_path)
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Load models
    sb_tag = "w" if subword else "wo"
    mlb = pkl.load(
        open(
            model_dir + "mlb_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    label_num = len(mlb.classes_)

    # Building Model
    print("Building computational graph...")
    if model_type == "BLSTM":
        print("Building default BLSTM mode with attention:", attention,
              "subword:", subword)
        model = BLSTM(
            label_num=label_num,
            sentence_emb=None,
            mention_emb=None,
            attention=attention,
            subword=subword,
            mode='concatenate',
            dropout=0.1)
    elif model_type == "CNN":
        print("Building default CNN mode with attention:", attention,
              "subword:", subword)
        model = CNN(
            label_num=label_num,
            sentence_emb=None,
            mention_emb=None,
            attention=attention,
            subword=subword,
            mode='concatenate',
            dropout=0.1)

    print(model.summary())

    file_path = model_type + "-weights-{epoch:02d}.hdf5"
    model_name = model_type + "-weights-00.hdf5"
    if attention:
        file_path = "Attention-" + file_path
        model_name = "Attention-" + model_name
    if subword:
        file_path = "Subword" + file_path
        model_name = "Subword" + model_name

    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        mode='min')  # Save every epoch
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early]  #early

    print("Loading testing data...")
    X = pkl.load(
        open(
            model_dir + "testing_data_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    X_m = pkl.load(
        open(
            model_dir + "testing_mention_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    y = pkl.load(
        open(
            model_dir + "testing_label_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))

    # Visualize number of rows
    test_size = 1000 if visualize else None
    model.load_weights(model_path)
    predict(
        model,
        X,
        X_m,
        y,
        model_file=model_path,
        output="results-test.txt",
        amount=test_size)
    """
    print("Loading file..")
    dataset = pd.read_csv("./data/smaller_preprocessed_sentence_keywords_labeled{0}.tsv"
                          .format("_subwords" if subword else ""),
                          sep='\t', names=['label','context','mention'])

    X = dataset['context'].values
    y = dataset['label'].values
    mentions = dataset['mention'].values
    train_index = pkl.load(open(model_dir + "train_index.pkl", 'rb'))
    test_index = pkl.load(open(model_dir + "test_index.pkl", 'rb'))
    X_train_text, X_test_text = X[train_index], X[test_index]
    X_train_mention_text, X_test_mention_text = mentions[train_index], mentions[test_index]
    del X, mentions
    """
    '''
    print("inverse_transform result...")
    y_pred = mlb.inverse_transform(y_pred)
    y_test_ = mlb.inverse_transform(y[:test_size])

    label_dict = json.load(open('data/label.json', "r"))
    inv_label_dict = {v: k for k, v in label_dict.items()}
    
    filename = trained_weight_file[:-5]+"_result.txt"

    print("Start output result...")
    if not visualize:
        with open(filename, "w") as f:
            for prediction, lab in tqdm(zip(y_pred, y_test_)):
                pred = list(prediction)
                pred.sort()
                label = list(lab)
                label.sort()
                pred = list(map(str, pred))
                pred = ",".join(pred)
                label = list(map(str, label))
                label = ",".join(label)
                f.write(pred + '\t')
                f.write(label + '\n')
                print("Writing file to", filename)
    else:
        for i in range(test_size):
            pred = list(y_pred[i])
            pred.sort()
            label = list(y_test_[i])
            label.sort()
            precision, recall, f1 = p_r_f(pred, label)
            pred = [inv_label_dict[j] for j in pred]
            label = [inv_label_dict[j] for j in label]
            temp = []
            temp.append(X_train_text[i])
            temp_ = []
            temp_.append(X_train_mention_text[i])
            if 'toxin' in temp_[0]:
                print("Sentence: ",temp)
                print("Mention",temp_)
                print("Prediction",pred)
                print("Label",label)
                score = "Precision: {:3.3f} Recall {:3.3f} F1 {:3.3f}".format(precision, recall, f1)
                print(score + "\n")
    '''
    K.clear_session()


def predict(model,
            X,
            X_m,
            y,
            model_file,
            output,
            amount=None,
            return_mf1=False,
            category=False):
    """
    Given the model object, data, labels, this function simply predict and evaluate
    the defined metrics with the model on the given data.

    Args:
        model(): Keras model object
        X(): Context
        X_m(): Mention
        y(): Targets labels
        model_file(str): Filename of the loaded weights
        output(): The filename of the evaluation metrics logging file.
        amount(int): The amount of first "amount" of data to be predicted.
        return_mf1(bool): Return micro-F1 score.
    """

    print("Predicting with saved model: {0} ... ".format(model_file), end='')
    start_time = time()
    y_pred = model.predict([X[:amount], X_m[:amount]])
    print("Done (took {:3.3f}s)".format(time() - start_time))

    print(y_pred)
    if category:
        y_pred = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)
    else:
        y_pred[y_pred >= 0.5] = 1.
        y_pred[y_pred < 0.5] = 0.

    print(y_pred)
    print(y_pred.sum(axis=1))
    y_pred = sparse.csr_matrix(y_pred)

    F1 = 0.
    file_writer = open(output, "a")
    file_writer.write("\n{0}\n".format(datetime.now()))
    file_writer.write("{:s}\n".format(model_file))

    # Begin evaluation
    print("Calculating Precision/Recall/F-1 scores ...")
    eval_types = ['micro', 'macro', 'weighted']
    for eval_type in eval_types:
        p, r, f, _ = precision_recall_fscore_support(
            y, y_pred, average=eval_type)
        print("[{}]\t{:3.3f}\t{:3.3f}\t{:3.3f}".format(eval_type, p, r, f))
        file_writer.write("[{}]\t{:3.3f}\t{:3.3f}\t{:3.3f}\n".format(
            eval_type, p, r, f))
        if eval_type == 'micro':
            F1 = f

    # Close file pointer
    file_writer.close()

    if return_mf1:
        return F1


def just_test(model,
              subword,
              filename,
              tag=None,
              postfix=None,
              amount=None,
              category=False):
    """
    Given the model object and the previously stored weights file,
    this function just restore the weights, load testing data and
    predict the labels.

    Args:
        model(): Keras model object.
        subword(bool): Indicating if the model uses subword information or not.
        filename(str): Filename of the trained weight file.
        amount(int): Use only first "amount" of data.
    """
    model_dir = "model/"
    sb_tag = "w" if subword else "wo"
    print("Restoring best weights from: {:s}".format(filename))
    model.load_weights(filename)

    # Load testing data
    print("Loading testing data...")
    X = pkl.load(
        open(
            model_dir + "testing_data_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    X_m = pkl.load(
        open(
            model_dir + "testing_mention_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))
    y = pkl.load(
        open(
            model_dir + "testing_label_{0}_subword_filter{1}.pkl".format(
                sb_tag, postfix), 'rb'))

    predict(
        model,
        X,
        X_m,
        y,
        model_file=filename,
        output="results-test.txt",
        amount=amount,
        category=category)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subword", action="store_true", help="Use subword or not")
    parser.add_argument(
        "--attention", action="store_true", help="Use attention or not")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize or not")
    parser.add_argument(
        "--model_dir",
        nargs='?',
        type=str,
        default="model/",
        help="Directory to load models. [Default: \"model/\"]")
    parser.add_argument(
        "--model_type",
        nargs='?',
        type=str,
        default="BLSTM",
        help="different model architecture BLTSM or CNN [Default: \"BLSTM/\"]")
    parser.add_argument(
        "--model_path",
        nargs='?',
        type=str,
        default="None",
        help="path to weighted")
    parser.add_argument(
        "--data_tag",
        nargs='?',
        type=str,
        help="Extra name tag on the dataset.")
    args = parser.parse_args()

    run(args.model_dir, args.model_type, args.model_path, args.subword,
        args.attention, args.data_tag, args.visualize)
