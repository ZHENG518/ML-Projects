import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(DIR)

import time
import torch
import json
import pandas as pd
import torch.optim as optim

from src import util
from src.model_GRU import Seq2seq
from src.dataset import QuestionSummaryDatasetloader, EN2CNDatasetloader


def get_config():
    config = {
        'n_epochs': 15,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'model_dim': 512,
        'n_layers': 3,
        'dropout': 0.5,
        'early_stop': 5,
        'save_path': f'{DIR}/models/summarisation.pth'
    }
    return config

def train(train_dataloader, valid_dataloader, model, word2ind, device, config):
    n_epochs = config['n_epochs']
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_record = {'train':[], 'dev':[]}
    early_stop_cnt, epoch, min_loss = 0, 0, float('inf')
    teacher_forcing_ratio_distribution = util.linear_distribution(num_epoch=n_epochs)

    while epoch < n_epochs:
        epoch_t0 = time.time()
        model.train()
        train_loss, train_acc, non_padding_sum = 0, 0, 0
        teacher_forcing_ratio = teacher_forcing_ratio_distribution[epoch]
        print(f'[Epoch{epoch+1}] teacher_forcing_ratio:{teacher_forcing_ratio}')
        for batch, (article, abstract) in enumerate(train_dataloader):
            batch_t0 = time.time()
            article_inds, article_padding_masks = util.x_seq2inds(word2ind, article)
            abstract_inds, abstract_padding_masks = util.y_seq2inds(word2ind, abstract)
            x = torch.LongTensor(article_inds).to(device)                        # (batch_size, max_len)
            y = torch.LongTensor(abstract_inds).to(device)                       # (batch_size, max_len)
            x_padding_masks = torch.BoolTensor(article_padding_masks).to(device) # (batch_size, max_len)
            y_padding_mask = torch.BoolTensor(abstract_padding_masks)            # (batch_size, max_len)
        # for batch, (x, y, x_padding_masks, y_padding_mask) in enumerate(train_dataloader):
        #     batch_t0 = time.time()
        #     x, y, x_padding_masks = x.to(device) , y.to(device) , x_padding_masks.to(device)
            non_padding = (y.shape[0]*(y.shape[1]-1))-y_padding_mask.sum().item()
            non_padding_sum += non_padding
            loss, acc, outputs = model.get_loss(x, y, x_padding_masks, y_padding_mask,
                                       teacher_forcing_ratio=teacher_forcing_ratio)
            train_loss += loss.detach().cpu().item()* non_padding
            train_acc += acc * non_padding

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            batch_t1 = time.time()
            torch.cuda.empty_cache()
            if (batch + 1) % 5 == 0:
                print('[Epoch{} Batch{}] loss:{:.3f} acc:{:.3f} time cost:{:.1f}s'.
                      format(epoch + 1, batch + 1, loss.item(), acc, batch_t1-batch_t0))

        train_loss = train_loss/non_padding_sum
        train_acc = train_acc/non_padding_sum
        print('Epoch: {:2d} Train_loss: {:.4f} Train_acc: {:.4f}'.format(epoch+1, train_loss, train_acc))

        valid_loss, valid_acc= valid(model, valid_dataloader, word2ind, device)

        epoch_t1 = time.time()
        print('Epoch: {:2d} Valid_loss: {:.4f} Valid_acc: {:.4f}'.
              format(epoch+1, valid_loss, valid_acc))
        print('Epoch Time Cost: {:.2f} mins'.format((epoch_t1-epoch_t0)/60))

        if valid_loss<min_loss:
            min_loss = valid_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch+1, min_loss))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1

def valid(model, valid_dataloader, word2ind, device):
    model.eval()
    valid_loss, valid_acc, non_padding_sum = 0, 0, 0
    for article, abstract in valid_dataloader:
        article_inds, article_padding_masks = util.x_seq2inds(word2ind, article)
        abstract_inds, abstract_padding_masks = util.y_seq2inds(word2ind, abstract)
        x = torch.LongTensor(article_inds).to(device)  # (batch_size, max_len)
        y = torch.LongTensor(abstract_inds).to(device)  # (batch_size, max_len)
        x_padding_masks = torch.BoolTensor(article_padding_masks).to(device)  # (batch_size, max_len)
        y_padding_mask = torch.BoolTensor(abstract_padding_masks)  # (batch_size, max_len)
    # for batch, (x, y, x_padding_masks, y_padding_mask) in enumerate(valid_dataloader):
    #     x, y, x_padding_masks = x.to(device) , y.to(device) , x_padding_masks.to(device)
        non_padding = (y.shape[0]*(y.shape[1]-1))-y_padding_mask.sum().item()
        non_padding_sum += non_padding
        with torch.no_grad():
            loss, acc, outputs = model.get_loss(x, y, x_padding_masks, y_padding_mask,
                                       teacher_forcing_ratio=-1)
        valid_loss += loss.detach().cpu().item() * non_padding
        valid_acc += acc * non_padding
        torch.cuda.empty_cache()
    valid_loss = valid_loss / non_padding_sum
    valid_acc = valid_acc / non_padding_sum

    return valid_loss, valid_acc

def test(model, test_dataloader,word2ind, ind2word, device, gen_max_len):
    model.eval()
    result = []
    for article in test_dataloader:
        article_inds, article_padding_masks = util.x_seq2inds(word2ind, article)
        x = torch.LongTensor(article_inds).to(device)  # (batch_size, max_len)
        x_padding_masks = torch.BoolTensor(article_padding_masks).to(device)  # (batch_size, max_len)
    # for x, _,x_padding_masks,_ in test_dataloader:
    #     x,x_padding_masks = x.to(device), x_padding_masks.to(device)
        pred = model(x, x_padding_masks, gen_max_len, word2ind['<BOS>'])
        # pred = model.single_beam_search(x, x_padding_masks, gen_max_len, word2ind, beam_size=3)
        for pred_result in pred:
            pred_words = []
            for ind in pred_result:
                if ind == word2ind['<EOS>']:
                    break
                pred_words.append(ind2word[str(int(ind))])
            result.append(pred_words)

    return result

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ =='__main__':

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('translation_GRU_coverage.txt')

    embedding = util.load_embedding(f'{DIR}/data/word_embedding.npy')
    embedding = torch.FloatTensor(embedding)
    word2ind = util.load_word2ind(f'{DIR}/data/vocab.txt')

    # en_word2ind = json.load(open(f'{DIR}/translation_data/cmn-eng/word2int_en.json'))
    # cn_word2ind = json.load(open(f'{DIR}/translation_data/cmn-eng/word2int_cn.json'))
    # word2ind = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config()

    model = Seq2seq(embedding_matrix=embedding,
                    padding_idx=word2ind['<PAD>'],
                    hidden_dim=config['model_dim'],
                    n_layers=config['n_layers'],
                    dropout=config['dropout'],
                    device=device,
                    coverage_weight=1,
                    use_pointer_gen=False,
                    ).to(device)
    #
    # model = Seq2seq(embedding_matrix=None,
    #                 padding_idx=en_word2ind['<PAD>'],
    #                 hidden_dim=config['model_dim'],
    #                 n_layers=config['n_layers'],
    #                 dropout=config['dropout'],
    #                 device=device,
    #                 coverage_weight=1,
    #                 use_pointer_gen = False,
    #                 en_vocab_size=len(en_word2ind),
    #                 cn_vocab_size=len(cn_word2ind),
    #                 emb_dim=256).to(device)

    train_dataloader = QuestionSummaryDatasetloader(f'{DIR}/data/train.csv', 'train', config['batch_size'])
    valid_dataloader = QuestionSummaryDatasetloader(f'{DIR}/data/valid.csv', 'valid', config['batch_size'])

    # train_dataloader = EN2CNDatasetloader(f'{DIR}/translation_data/cmn-eng', 50, 'training', config['batch_size'])
    # valid_dataloader = EN2CNDatasetloader(f'{DIR}/translation_data/cmn-eng', 50, 'validation', config['batch_size'])
    # model.load_state_dict(torch.load(f'{DIR}/models/translation_GRU_coverage.pth', map_location=torch.device(device)))

    train(train_dataloader, valid_dataloader, model, word2ind, device, config)
    # valid_loss, valid_acc = valid(model, valid_dataloader, word2ind, device)
    # print('Valid_loss: {:.4f} Valid_acc: {:.4f}'.format(valid_loss, valid_acc))

    # model.load_state_dict(torch.load(f'{DIR}/models/translation_GRU_coverage.pth', map_location=torch.device(device)))
    # cn_ind2word = json.load(open(f'{DIR}/translation_data/cmn-eng/int2word_cn.json'))
    # test_dataloader = EN2CNDatasetloader(f'{DIR}/translation_data/cmn-eng', 50, 'testing', config['batch_size'])
    # pred_result = test(model, test_dataloader, en_word2ind, cn_ind2word, device, 50)
    #
    #
    # with open(f'{DIR}/translation_data/cmn-eng/testing.txt', 'r') as f:
    #     lines = f.readlines()
    # with open(f'{DIR}/translation_data/cmn-eng/pred_result.txt', 'w') as f:
    #     for ind, line in enumerate(lines):
    #         data = line.strip().split('\t')
    #         data.append(' '.join(pred_result[ind]))
    #         f.write('\t'.join(data)+'\n')