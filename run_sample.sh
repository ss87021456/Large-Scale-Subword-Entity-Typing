set -x
CUDA_VISIBLE_DEVICES=5 python ./src/train.py --context_emb=../word2vec/kbp_part_100.emb --context_embedding_dim=100 --arch=cnn --data_tag=kbp --use_softmax --tag=100 --data_tag=kbp --attention
# --mention_emb
# --mention_embedding_dim

