import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def load_word2ind(vocab_path):
    word2ind = {}
    with open(vocab_path, 'r') as f:
        for ind, word in enumerate(f.readlines()):
            word2ind[word.strip('\n')] = ind
    return word2ind

def load_ind2word(vocab_path):
    ind2word = []
    with open(vocab_path, 'r') as f:
        for word in f.readlines():
            ind2word.append(word.strip('\n'))
    return ind2word

def x_seq2inds(word2ind, seqs):
    unk_ind = word2ind['<UNK>']
    pad_ind = word2ind['<PAD>']
    len_list = [len(seq) for seq in seqs]
    max_len =  min(128,max(len_list))
    # max_len = int(np.mean(len_list) + 2 * np.std(len_list))
    inds = []
    padding_masks = []
    for seq in seqs:
        words = seq.split()[:max_len]
        # 转换为ind
        ind = [word2ind.get(word,unk_ind) for word in words]
        # padding mask
        padding_mask = [0] * len(ind)
        # 添加<PAD>
        padding_len = max_len - len(ind)
        ind += [pad_ind] * padding_len
        padding_mask += [1]*padding_len

        inds.append(ind)
        padding_masks.append(padding_mask)
    return inds, padding_masks

def y_seq2inds(word2ind, seqs):
    unk_ind = word2ind['<UNK>']
    pad_ind = word2ind['<PAD>']
    eos_ind = word2ind['<EOS>']
    bos_ind = word2ind['<BOS>']
    len_list = [len(seq) for seq in seqs]
    max_len = min(128,max(len_list))
    # max_len = int(np.mean(len_list) + 2 * np.std(len_list))
    inds = []
    padding_masks = []
    for seq in seqs:
        words = seq.split()[:max_len]
        # 添加<BOS>
        ind = [bos_ind]
        # 转换为ind
        ind += [word2ind.get(word,unk_ind) for word in words]
        # 添加<EOS>
        ind.append(eos_ind)
        padding_mask = [0] * len(ind)
        # 添加<PAD>
        padding_len = max_len+2 - len(ind)
        ind += [pad_ind] * padding_len
        padding_mask += [1]*padding_len

        inds.append(ind)
        padding_masks.append(padding_mask)
    return inds, padding_masks

def load_embedding(path):
    embedding = np.load(path)
    return embedding

def linear_distribution(num_epoch):
  return np.linspace(start=1,stop=0.0,num=num_epoch,dtype=np.float32)

def bleu_score(reference, predict):
    score = 0
    for ref, pred in zip(reference, predict):
        score += sentence_bleu([ref], pred, weights=(1,0,0,0))
    return score