echo "Generating training, testing index..."
python ./src/generate_index.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv 

echo "Generating pickle files for training, testing w/o subword..."
python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv

echo "Generating pickle files for training, testing w subword..."
python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled_subwords.tsv --subword
