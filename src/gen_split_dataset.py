import argparse


def gen_split_dataset():
    """
    Generate and split dataset according to the given criteria

    Arguments:

    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Raw dataset.")
    parser.add_argument("rule", help="Rules for vocabulary list to be cleaned up.")
    # optional arguments
    parser.add_argument("--validation", nargs='?', const=0.1, type=float,
                        help="The ratio of validation dataset.")
    parser.add_argument("--testing", nargs='?', const=0.1, type=float,
                        help="The ratio of testing dataset.")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores available.")
    args = parser.parse_args()

    gen_split_dataset()