# Generate training, testing pickle file
echo "Generating mention split data 90% training, 10% testing.."
python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv

# Start train the model
echo "Start Train the model.."
CUDA_VISIBLE_DEVICES=0 python ./src/BLSTM_train.py --pre=False

