set -x
echo ""
echo "[STAGE 0/4] install nltk stopwords, punkt package"
python src/nltk_download.py

# Prepare keyword
echo ""
echo "[STAGE 1/4] Parsing and trimming infrequent labels in given hierarchy tree, parsing subwords information"
python src/parse_entity.py data/MeSH_type_hierarchy.txt --trim --threshold=1
python src/parse_entity.py data/UMLS_type_hierarchy.txt --trim --threshold=1
python src/parse_entity.py data/custom_subwords_v2.txt --subword

# Preliminary
echo ""
echo "[STAGE 2/4] Preliminary cleanup the corpus"
python src/preliminary.py data/smaller.tsv src/refine_rules/preliminary.tsv --thread=5

# Recognize sentences
echo ""
echo "[STAGE 3/4] Recognize all sentences which contain mentions in given entity hierarchy tree in the given corpus"
python src/recognize_sentences.py data/smaller_preprocessed.tsv data/ --trim --mode=MULTI --thread=20

# add labels to the dataset
echo ""
echo "[STAGE 4/4] Adding labels to the dataset"
echo "python src/label.py data/"
python src/label.py data/ --trim
python src/label.py data/ --labels=data/label.json --replace --corpus=data/smaller_preprocessed_sentence.tsv --thread=10
python src/label.py data/ --labels=data/label.json --replace --corpus=data/smaller_preprocessed_sentence.tsv --subwords=data/subwords.json --thread=10
