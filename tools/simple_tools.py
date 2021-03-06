import argparse
import json
import os
"""
Extract content to ascii (for training embedding) [column is specified by --position]
python tools/simple_tools.py ../share/data.txt ../share/kbp_content_only.txt --position=1 --extract_col --to_ascii

Convert non-ASCII encoding text to ASCII one
python tools/simple_tools.py ../share/data.txt ../share/kbp_ascii.tsv --to_ascii

Parse wordnet description
python tools/simple_tools.py types_with_entity_mapping.json wordnet_desc.json --parse_desc --text_only

Check keys presents in the collection
python tools/simple_tools.py data/label_kbp.json wordnet_desc.json --check_key
"""


def extract_col(args):
    """
    Extract content from specified column in the .tsv file.
    Duplicated lines will be eliminated.
    """
    count = 0
    n_duplicate = 0
    data = list()
    for line in open(args.file, "r"):
        count += 1
        tmp = line.split("\t")[args.position]
        # Filter out duplicated lines (same as previous line)
        # Error occur when the list is empty (first time)
        try:
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


def asASCII(content):
    return content.encode("ascii", "replace").decode("ascii")


def write_as_ascii(args, data, delimitor=None):
    with open(args.output, "w", encoding='ascii') as f:
        for content in data:
            if delimitor is not None:
                "".join(content.split("\t"))

            f.write("{0}\n".format(asASCII(content) if args.to_ascii else content))
    pass


def to_ascii(args):
    with open(args.file, "r") as f:
        data = f.read().splitlines()
    write_as_ascii(args, data)


def parse_description(args):
    """
    Deal with improper format of json file with each line coresponds to one
    entry of word (one dict per line.)

    Wrap entire file inside [ ] and add comma at the end of each line to
    form a loadable json file (load as list of dicts)

    Then for each dicts in the list, use "wordnet" as their key and
    ["wordnet" + "definition"] as values to generate a new dict with word
    as key and their definition as values

    Note: Some entry might have no definition (None/Null,) currently we add
    the word itself in the beginning of the definition, i.e. [word] [def],
    to temporarily circumvent this issue.
    """
    print("Parsing wordnet description file from: {}".format(args.file))
    with open(args.file, "r") as f:
        raw = f.read().splitlines()
    raw_len = len(raw)
    print(" * {} entries (lines) read from file".format(raw_len))

    # Save to temporary file
    tmp_file = "tmp.json"
    with open(tmp_file, "w") as f:
        f.write("[\n")
        # Add comma at the end of each line except the last line.
        for itr in raw[:-1]:
            content = asASCII(itr) if args.to_ascii else itr
            f.write(content + ",\n")
        # Add last line and left bracket
        content = asASCII(raw[-1]) if args.to_ascii else raw[-1]
        f.write(content + "\n]\n")
    print(" - Saved to temporary file: {}".format(tmp_file))
    del raw

    # Reload the parsed version of the json file
    print("Loading parsed dictionary from temporary file...")
    raw = json.load(open(tmp_file, "r"))
    # Keep temporary file for debugging if needed
    if not args.save_tmp:
        os.remove(tmp_file)
        print(" - Removed temporary file: {}".format(tmp_file))

    # Dictionary with word as keys and description as values
    print("Converting temporary file to proper word-definition format")
    keys = [itr["wordnet"].lower() for itr in raw]
    vals = [{"definition": "{} {}"
            .format(itr["wordnet"], itr["defintion"])} for itr in raw]
    # vals = [{"definition": itr['defintion'], "entities": itr["entities"]} for itr in raw]

    dic = dict(zip(keys, vals))
    assert len(dic) == raw_len
    print(" * {} entries created".format(len(dic)))

    # Dump dictionary to file
    with open(args.output, "w") as f:
        json.dump(dic, f, sort_keys=True, indent=4)
    print("Parsed file save in file: {}".format(args.output))
    del dic


def check_key_in_collection(args):
    """
    Check if the keys in the first given file all present in the second
    file to check if key-error exists in the collection (second file)
    """
    label = json.load(open(args.file))
    collection = json.load(open(args.output))

    print("Given json information:")
    print(" - Label: {} entries".format(len(label)))
    print(" - Collection: {} entries\n".format(len(collection)))

    check = [(itr in collection) for itr in list(label.keys())]
    absent = len(label) - sum(check)
    print("Result:")
    print(" - {} keys are absent in given collection ({})".format(
        absent, "PASSED" if absent == 0 else "FAILED"))
    print(" - {} keys are presented in given collection".format(sum(check)))
    print(" * {} keys are absent in given label".format(
        len(collection) - sum(check)))


def main(args):
    if args.extract_col:
        extract_col(args)
    elif args.to_ascii:
        to_ascii(args)
    elif args.parse_desc:
        parse_description(args)
    elif args.check_key:
        check_key_in_collection(args)
    else:
        print("No job to be done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the target file.")
    parser.add_argument("output", type=str, help="Output filename.")
    parser.add_argument(
        "--task", help="The specific task to perform on the file.")
    parser.add_argument(
        "--parse_desc", action="store_true", help="Parse wordnet description.")
    parser.add_argument(
        "--extract_col", action="store_true", help="Extract column from file.")
    parser.add_argument(
        "--to_ascii", action="store_true", help="Convert input file to ASCII.")

    parser.add_argument(
        "--save_tmp", action="store_true", help="Save temporary file.")

    parser.add_argument(
        "--check_key",
        action="store_true",
        help="Check keys in two json files.")

    parser.add_argument(
        "--position", type=int, help="The index of the column in interest.")

    args = parser.parse_args()

    main(args)
