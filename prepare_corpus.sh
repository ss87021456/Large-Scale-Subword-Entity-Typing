# Prepare keyword
echo ""
echo "[STAGE 1/5] Parsing and trimming infrequent labels in given hierarchy tree, parsing subwords information"
python src/parse_entity.py data/MeSH_type_hierarchy.txt --trim --threshold=1
python src/parse_entity.py data/UMLS_type_hierarchy.txt --trim --threshold=1
python src/parse_entity.py data/custom_subwords_v2.txt --subword

# Preliminary
echo ""
echo "[STAGE 2/5] Preliminary cleanup the corpus"
python src/preliminary.py data/smaller.tsv src/refine_rules/preliminary.tsv --thread=5

# Parse sentences'
echo ""
echo "[STAGE 3/5] Parse and split the sentences in the corpus"
python src/parse_sentence.py data/smaller_preprocessed.tsv --thread=10

# Recognize sentences
echo ""
echo "[STAGE 4/5] Recognize all sentences in the corpus containing the mentions in given entity hierarchy tree"
python src/recognize_sentences.py data/smaller_preprocessed_sentence.txt data/ --trim --mode=MULTI --thread=20

# add labels to the dataset
echo ""
echo "[STAGE 5/5] Adding labels to the dataset"
python src/label.py data/
python src/label.py data/ --labels=data/label.json --replace --corpus=data/smaller_preprocessed_sentence_keywords.tsv --thread=10
python src/label.py data/ --labels=data/label.json --replace --corpus=data/smaller_preprocessed_sentence_keywords.tsv --subwords=data/subwords.json --thread=10

