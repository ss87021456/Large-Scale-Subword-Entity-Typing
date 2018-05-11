echo "Generating training, testing index..."
python ./src/generate_index.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv 

echo "Generating pickle files for training, testing w/o subword..."
python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --train_idx=./model/train_index.pkl --test_idx=./model/test_index.pkl --vali_idx=./model/validation_index.pkl

echo "Generating pickle files for training, testing w subword..."
python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled_subwords.tsv --subword --train_idx=./model/train_index.pkl --test_idx=./model/test_index.pkl --vali_idx=./model/validation_index.pkl
