import argparse


def gen_split_dataset(file, keywords, mode, validation, testing, thread):
    """
    Generate and split dataset according to the given criteria

    Arguments:
        file(str):
        keywords(str):
        mode(str):
        validation(float):
        testing(float):
        thread(str):

    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Raw dataset.")
    # optional arguments
    parser.add_argument("--keywords", const="data/keywords.json", \
                        help="Path to keyword dictionary")
    parser.add_argument("--mode", choices=["SINGLE", "MULTI"], \
                        const="SINGLE", help="Single mention or \
                        multi-mentions per sentence.")
    parser.add_argument("--validation", nargs='?', const=0.1, type=float,
                        help="The ratio of validation dataset.")
    parser.add_argument("--testing", nargs='?', const=0.1, type=float,
                        help="The ratio of testing dataset.")
    parser.add_argument("--thread", type=int, help="Number of threads \
                        to run, default: 2 * number_of_cores available.")
    args = parser.parse_args()

    gen_split_dataset(args.file, args.keywords, args.mode, args.validation,
                      args.testing, args.thread)
