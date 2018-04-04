python ./src/generate_data.py --input=./data/smaller_preprocessed_sentence_keywords_labeled.tsv
CUDA_VISIBLE_DEVICES=6,7 python ./src/BLSTM_train.py --corpus=./data/smaller_preprocessed_sentence_keywords_labeled.tsv --pre=True --emb=./data/model.vec --evaluation

