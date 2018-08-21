set -x
CUDA_VISIBLE_DEVICES=7 python ./src/train.py --context_embedding_dim=100 --mention_embedding_dim=100 --arch=cnn --data_tag=kbp --use_softmax --data_tag=kbp --indicator --learning_rate=0.005 --epochs=5 --tag=1
# --mention_emb
# --mention_embedding_dim

