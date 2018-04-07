# Preliminary
echo "Preliminary cleanup the corpus"
python src/preliminary.py data/smaller.tsv src/refine_rules/preliminary.tsv --thread=10

# Parse sentences
echo "Parse and split the sentences in the corpus"
python src/parse_sentence.py data/smaller_preprocessed.tsv --thread=10

# Recognize sentences
echo "Recognize all sentences in the corpus containing the mentions in given entity hierarchy tree"
python src/recognize_sentences.py data/smaller_preprocessed_sentence.txt data/ --thread=20

# add labels to the dataset
echo "Adding labels to the dataset"
python src/label.py data/keywords.json --labels=data/label.json --replace --corpus=data/smaller_preprocessed_sentence_keywords.tsv --thread=10
