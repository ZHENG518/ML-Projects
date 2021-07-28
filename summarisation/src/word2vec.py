import pandas as pd
import numpy as np
from gensim.models import word2vec

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

texts = train_df.input.to_list()+train_df.report.to_list()+test_df.input.to_list()
texts = [line.split() for line in texts]

model = word2vec.Word2Vec(texts, vector_size=256, window=5,
                          sorted_vocab=1,
                          min_count=5, # Ignores all words with total frequency lower than this.
                          epochs=5,    # Number of iterations (epochs) over the corpus.
                          sg=1)        # Training algorithm: 1 for skip-gram; otherwise CBOW.

index2word = model.wv.index_to_key
word_embedding = model.wv.vectors

# 添加<BOS><EOS><PAD>到vocab和embedding matrix
index2word = index2word+['<BOS>','<EOS>','<PAD>','<UNK>']
vector = np.random.randn(4,model.vector_size) # 生成随机的四个vector
embedding_matrix = np.vstack((word_embedding, vector))

# 保存vocab
with open('../data/vocab.txt', 'w') as f:
    for word in index2word:
        f.write(word+'\n')

# 保存word embedding
np.save('../data/word_embedding.npy', embedding_matrix)
