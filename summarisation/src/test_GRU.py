import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(DIR)

import torch
from src.dataset import QuestionSummaryDatasetloader
from src import util
from src.model_GRU import Seq2seq

gen_max_len = 125

embedding = util.load_embedding(f'{DIR}/data/word_embedding.npy')
embedding = torch.FloatTensor(embedding)
word2ind = util.load_word2ind(f'{DIR}/data/vocab.txt')
ind2word = util.load_ind2word(f'{DIR}/data/vocab.txt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = get_config()

test_dataloader = prep_dataloader(f'{DIR}/data/test.csv', 'test', batch_size=32)
model = Seq2seq(embedding_matrix=embedding,
                padding_idx=word2ind['<PAD>'],
                hidden_dim=config['model_dim'],
                dropout=config['dropout'],
                device=device,
                coverage_weight=0).to(device)
model.load_state_dict(torch.load(f'{DIR}/models/GRU_v2.pth', map_location=torch.device(device)))
model.eval()

result = []
for article in test_dataloader:
    article_inds, article_padding_masks = util.x_seq2inds(word2ind, article)
    x = torch.LongTensor(article_inds).to(device)  # (batch_size, max_len)
    x_padding_masks = torch.BoolTensor(article_padding_masks).to(device)  # (batch_size, max_len)

    pred = model(x, x_padding_masks, gen_max_len, word2ind['<BOS>'])
    # pred = model.single_beam_search(x, x_padding_masks, gen_max_len, word2ind, beam_size=3)
    for pred_result in pred:
        words = [ind2word[int(ind)] for ind in pred_result]
        result.append(words)

print('hhh')