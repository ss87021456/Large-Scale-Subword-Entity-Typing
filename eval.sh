# Generate training, testing pickle file
echo "Generating mention split data 90% training, 10% testing base on mention..."
python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv \
 --train_idx=./model/train_index.pkl --test_idx=./model/test_index.pkl
echo "Generating subword version training, testing data..."
python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled_subwords.tsv \
 --subword --train_idx=./model/train_index.pkl --test_idx=./model/test_index.pkl

# Start train the model
echo "Start Train the model.."
CUDA_VISIBLE_DEVICES=1 python ./src/train.py --pre=True --emb=./data/4_20_embedding.vec --mode=BLSTM

echo "Output Testing prediction..."
CUDA_VISIBLE_DEVICES=1 python ./src/test.py --model_path=BLSTM-05.hdf5 --model_type=BLSTM

echo "Evaluate layer accuracy..."
python src/hierarchical_eval.py --labels=data/label.json \
--mention=test_mention_list.txt --prediction=BLSTM-05_result.txt \
--k_parents=13 --hierarchy=data/
