set -x
CUDA_VISIBLE_DEVICES=5 python ./src/train.py --context_embedding_dim=300 --mention_embedding_dim=100 --arch=cnn --data_tag=kbp --use_softmax --tag=c3m1 --data_tag=kbp --learning_rate=0.005 --epochs=7
# --mention_emb
# --mention_embedding_dim

