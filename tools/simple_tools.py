import argparse
"""
Extract content to ascii (for training embedding) [column is specified by --position]
python tools/simple_tools.py ../share/data.txt ../share/kbp_content_only.txt --position=1 --extract_col --to_ascii

Convert non-ASCII encoding text to ASCII one
python tools/simple_tools.py ../share/data.txt ../share/kbp_ascii.tsv --to_ascii

"""


def extract_col(args):
    count = 0
    n_duplicate = 0
    data = list()
    for line in open(args.file, "r"):
        count += 1
        tmp = line.split("\t")[args.position]
        # print(tmp)
        # print(tmp.encode("utf-8").decode("ascii"))
        # exit()
        try:
            # Filter out duplicated lines
            # Error occur when the list is empty (first time)
            if tmp == data[-1]:
                n_duplicate += 1
                continue
        except:
            pass
        # Add valid line to list
        data.append(tmp)
    print("Total lines in file: {:d}".format(count))
    print(" - {:d} duplicated lines ({:2.2f}%)".format(
        n_duplicate, 100. * n_duplicate / count))

    write_as_ascii(args, data)


def write_as_ascii(args, data, delimitor=None):
    with open(args.output, "w", encoding='ascii') as f:
        for content in data:
            if delimitor is not None:
                "".join(content.split("\t"))
            # print(repr(content))
            f.write("{0}\n".format(
                content.encode("ascii", 'replace').decode("ascii")
                if args.to_ascii else content))
    pass


def to_ascii(args):
    with open(args.file, "r") as f:
        data = f.read().splitlines()
    write_as_ascii(args, data)


def main(args):
    if args.extract_col:
        extract_col(args)
    elif args.to_ascii:
        to_ascii(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the target file.")
    parser.add_argument("output", type=str, help="Output filename.")
    parser.add_argument(
        "--task", help="The specific task to perform on the file.")
    parser.add_argument(
        "--extract_col", action="store_true", help="Extract column from file")
    parser.add_argument(
        "--to_ascii", action="store_true", help="Convert input file to ASCII.")

    parser.add_argument(
        "--position", type=int, help="The index of the column in interest.")

    args = parser.parse_args()

    main(args)
