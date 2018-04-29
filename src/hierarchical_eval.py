import argparse
import json
import numpy as np
from itertools import chain
from utils import write_to_file, readlines, merge_dict

# python src/hierarchical_eval.py --labels=data/label.json \
# --mention=test_mention_list.txt --prediction=sample_pred_label.txt \
# --k_parents=5 --hierarchy=data/


def k_parents_eval(labels, mention, pred_file, k_parents, hierarchy_path, output=None):
    """
    """
    eval_mention = readlines(mention)
    # eval_mention = list(chain.from_iterable(eval_mention))

    #
    hierarchy_dict = merge_dict(hierarchy_path, postfix="_kptree.json")
    # data/label.json
    # data/label_lookup.json
    # Load label mapping
    labels_dict = json.load(open(labels, "r"))
    # Reverse label mapping to text content
    labels_dict = {val: key for key, val in labels_dict.items()}

    # Load result for eval
    pred = readlines(pred_file)
    pred = [itr.split("\t") for itr in pred]
    pred = [[itr[0].split(","), itr[1].split(",")] for itr in pred]

    # Parse str to integer and lookup for the type (in context)
    print("\nCalculating {:d} Layer parents accuracy:".format(k_parents))
    acc_collection = list()
    depth = list()
    for idx, itr in enumerate(pred):
        prediction, ground_truth = itr
        paths = hierarchy_dict[eval_mention[idx]]
        depth.append([len(itr) for itr in paths])
        # Lookup the dictionary containing the path of each node
        # Each entry contains list(All paths) of list(Path)
        prediction = [-1 if itr == "" else labels_dict[int(itr)] for itr in prediction]
        ground_truth = [labels_dict[int(itr)] for itr in ground_truth]
        # print("PATH: {0}".format(paths))
        # print("{0}: {1} | {2}".format(idx + 1, prediction, ground_truth))
        tmp = list()
        for itr_path in paths:
            per_path_acc = list()
            for itr_k in range(k_parents):
                # print("Layer {:d}: {:s}".format(itr_k, itr_path[itr_k]))
                try:
                    # tmp.append(1. if itr_path[itr_k] in prediction else 0.)
                    per_path_acc.append(1. if itr_path[itr_k] in prediction else 0.)
                # Path is not long enough
                except:
                    # tmp.append(np.nan)
                    per_path_acc.append(-1.)
            # print(tmp)
            # acc_collection.append(tmp)
            tmp.append(per_path_acc)

        # If the prediction predict correctly on some layer k, we give it 100% accuracy
        tmp = np.array(tmp)
        tmp = tmp.max(axis=0)
        tmp[tmp == -1.] = np.nan
        acc_collection.append(tmp)

    # Calculate accuracy with respect to batch and ignore NaNs
    acc_collection = np.array(acc_collection)
    accuracy = np.nanmean(acc_collection, axis=0)
    # Calculate number of instance across all classes
    n_instances = np.count_nonzero(~np.isnan(acc_collection), axis=0)
    # print(accuracy)
    print(" LAYER# | N_INSTANCES | ACCURACY")
    print("================================")
    for itr, n in zip(range(k_parents), n_instances):
        print("Layer {:2d}:  {:10d} |   {:.2f}%".format(itr + 1, n, 100. * accuracy[itr]))
    # info
    depth = np.array(list(chain.from_iterable(depth)))
    print("================================")
    print("Depth: MEAN={:2.2f} | MAX={:2.2f} | MIN={:2.2f}"
          .format(depth.mean(), depth.max(), depth.min()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", help="Label for mapping types to label values.")
    parser.add_argument("--mention", help="Mention of each instance, for evaluation.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--prediction", help="Dumped prediction.")
    parser.add_argument("--k_parents", type=int, default=5, help="Dumped prediction.")
    parser.add_argument("--hierarchy", help="Hierarchy information for k-parent evaluation.")

    args = parser.parse_args()

    k_parents_eval(args.labels, args.mention, args.prediction, args.k_parents,
                   args.hierarchy, args.output)