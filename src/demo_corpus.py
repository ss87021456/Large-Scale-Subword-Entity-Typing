from utils import readlines, write_to_file
import argparse
"""
python src/demo_corpus.py [file] --rand --amount=10

"""


def sample_data_for_demo(files, amount, rand):
    """
    Make sample of the large dataset for demo on online platform

    Arguments:
        files(list of str): Path to the files to be sampled.
        amount(int): Number of first "amount" of lines are sampled.

    """
    print("Total number of files: {0}".format(len(files)))
    for itr in files:
        output = itr[:-4] + "_sampled" + itr[-4:]
        data = readlines(itr, limit=amount, rand=rand)
        write_to_file(output, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs='+', type=str, help="Input files to be sampled.")
    parser.add_argument(
        "--amount",
        nargs='?',
        type=int,
        const=10,
        help="Number of samples drawn from the files.")
    parser.add_argument(
        "--rand", action="store_true", help="Randomly draw samples from file.")
    args = parser.parse_args()

    sample_data_for_demo(args.files, args.amount, args.rand)
