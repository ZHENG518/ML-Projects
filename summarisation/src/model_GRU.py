import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Seq2seq(nn.Module):
    def __init__(self,embedding_matrix, padding_idx, hidden_dim, n_layers, dropout, device,
                 coverage_weight=1.0,use_pointer_gen=False,en_vocab_size=None, cn_vocab_size=None, emb_dim=None):
        super().__init__()
        self.device = device
        self.coverage_weight = coverage_weight
        self.use_coverage = coverage_weight > 0
        self.encoder = Encoder(embedding_matrix, padding_idx, hidden_dim, n_layers, dropout, en_vocab_size=en_vocab_size, emb_dim=emb_dim)
        self.decoder = Decoder(embedding_matrix, padding_idx, hidden_dim, n_layers, dropout,
                               use_pointer_gen=use_pointer_gen, use_coverage=self.use_coverage,
                               cn_vocab_size=cn_vocab_size, emb_dim=emb_dim)
        self.criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=padding_idx)

    def get_loss(self, x, y, x_padding_mask, y_padding_mask, teacher_forcing_ratio):
        batch_size = x.shape[0]
        input_length = x.shape[1]
        target_length = y.shape[1]-1
        non_pad_num = (batch_size*target_length)-y_padding_mask.sum().item()

        enc_outputs, enc_hiddens = self.encoder(x)
        # enc_outputs(batch_size, length, hidden_dim*2) 每个时刻的输出
        # enc_hiddens(2*nun_layers, batch_size, hidden_dim) 最后时刻的输出
        # enc_hiddens =  [2 * nun_layers, batch size  , hid dim]  --> [num_layers, directions, batch size, hid dim]
        enc_hiddens = enc_hiddens.view(self.encoder.n_layers, 2, batch_size, -1)
        # 把两个方向的encoder hidden拼在一起 [num_layers, batch size, hid dim*2]
        enc_hiddens = torch.cat((enc_hiddens[:, -2, :, :], enc_hiddens[:, -1, :, :]), dim=2)

        dec_hiddens = enc_hiddens           # 使用encoder的hidden作为t0时刻的decoder hidden
        dec_input = y[:, 0].unsqueeze(1)    # (batch_size, 1) 0时刻decoder输入为<BOS>
        loss = 0
        acc = 0
        coverage = torch.zeros(batch_size, input_length).to(self.device) if self.use_coverage else None
        outputs = []
        for i in range(1, y.shape[1]):
            dec_target = y[:,i]
            target_padding_mask = y_padding_mask[:,i]
            predictions, dec_hiddens, attention_weights, new_coverage = self.decoder(dec_input, enc_outputs, dec_hiddens,
                                                                                     x_padding_mask, coverage)
            # predictions (batch_size, vocab_size)
            # dec_hiddens (batch_size, hidden_dim)
            pred = torch.argmax(predictions, dim=1)
            outputs.append(pred.data.tolist())
            loss_t = self.criterion(input=predictions, target=dec_target)
            if self.use_coverage:
                coverage_loss = torch.sum(torch.min(attention_weights, coverage))
                loss_t += coverage_loss
                coverage = new_coverage
            loss += loss_t
            pred = torch.argmax(predictions, dim=1).cpu()      # pred (batch_size)
            acc += torch.eq(pred, dec_target.cpu()).int().masked_fill_(target_padding_mask, 0).sum().item()
            if random.random() <= teacher_forcing_ratio: # random()在[0.0, 1.0)范围内生成随机数, ratio [0.0, 1.0]随着epoch增加而减小
                dec_input = y[:, i].unsqueeze(1)
            else:
                dec_input = pred.unsqueeze(1).to(self.device)
        loss = loss/(batch_size*target_length)
        acc = acc/non_pad_num
        outputs = np.array(outputs).T
        return loss, acc, outputs

    def forward(self, x, x_padding_mask, gen_max_len, bos_idx):
        batch_size = x.shape[0]
        input_length = x.shape[1]
        # encode input
        enc_outputs, enc_hiddens = self.encoder(x)
        enc_hiddens = enc_hiddens.view(self.encoder.n_layers, 2, batch_size, -1)
        # 把两个方向的encoder hidden拼在一起 [num_layers, batch size, hid dim*2]
        enc_hiddens = torch.cat((enc_hiddens[:, -2, :, :], enc_hiddens[:, -1, :, :]), dim=2)

        # decoder t0 input
        dec_hiddens = enc_hiddens
        dec_input = torch.LongTensor([bos_idx]*batch_size).unsqueeze(1).to(self.device)
        # predict result
        outputs = np.zeros([gen_max_len, batch_size])
        coverage = torch.zeros(batch_size, input_length).to(self.device) if self.use_coverage else None
        for i in range(gen_max_len):
            predictions, dec_hiddens, attention_weights, coverage = self.decoder(dec_input, enc_outputs, dec_hiddens,
                                                                    x_padding_mask, coverage)
            pred = torch.argmax(predictions, dim=1)
            outputs[i] = pred.cpu()
            dec_input = pred.unsqueeze(1)

        outputs = outputs.T
        return outputs

    def single_beam_search(self, x, x_padding_mask, gen_max_len, word2ind, beam_size):
        """把每一步的k个可能当作一个batch送进decoder"""
        bos_idx = word2ind['<BOS>']
        eos_idx = word2ind['<EOS>']
        vocab_size = len(word2ind)

        enc_outputs, enc_hiddens = self.encoder(x)

        k_prev_words = torch.full((beam_size, 1), bos_idx, dtype=torch.long).to(self.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(beam_size, 1).to(self.device)

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        # 第一步的k种可能是相同的
        dec_hiddens = enc_hiddens.squeeze().repeat(beam_size,1).unsqueeze(1)
        enc_outputs = enc_outputs.squeeze().repeat(beam_size,1,1)
        while True:
            predictions, dec_hiddens = self.decoder(k_prev_words, enc_outputs, dec_hiddens, x_padding_mask)
            # predictions (beam_size, vocab_size)
            # dec_hiddens (beam_size, 1, hidden_dim)
            pred_log_probs = F.log_softmax(predictions, dim=1)
            cumulative_log_probs = pred_log_probs+top_k_scores.repeat(1, vocab_size)
            if step == 1:
                # 因为最开始解码的时候只有一个结点<BOS>,所以只需要取其中一个结点计算topk
                top_k_scores, top_k_words = cumulative_log_probs[0].topk(beam_size, largest=True, sorted=True)
            else:
                # 将k*vocab_size的矩阵展开，就代表每个beam的预测结果
                top_k_scores, top_k_words = cumulative_log_probs.view(-1).topk(beam_size, 0, True, True)

            prev_word_inds = top_k_words // vocab_size  # (k)  实际是beam_id
            next_word_inds = top_k_words % vocab_size  # (k)  实际是token_id
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            # 当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置, 实际是beam_id)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != eos_idx]
            # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())  # 加入句子
                complete_seqs_scores.extend(top_k_scores[complete_inds])  # 加入句子对应的累加log_prob
                # 减掉已经完成的句子的数量，更新k, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
                beam_size -= len(complete_inds)

            seqs = seqs[incomplete_inds]
            dec_hiddens = dec_hiddens[prev_word_inds[incomplete_inds]]
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

            if step == gen_max_len:
                break
            step += 1


class Encoder(nn.Module):
    def __init__(self,embedding_matrix, padding_idx, hidden_dim, n_layers, dropout, en_vocab_size=None, emb_dim=None):
        super().__init__()
        self.n_layers = n_layers
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=padding_idx)
            self.gru = nn.GRU(input_size=embedding_matrix.shape[1],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            self.embedding = nn.Embedding(en_vocab_size, emb_dim)
            self.gru = nn.GRU(input_size=emb_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, hidden= self.gru(x)
        # outputs(batch_size, length, 2*hidden_dim) 每个时刻的输出
        # hiddens(2*nun_layers, batch_size, hidden_dim) 最后时刻的输出
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, use_coverage):
        super().__init__()
        self.value_layer = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.query_layer = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.energy_layer = nn.Linear(hidden_dim, 1, bias=False)
        if use_coverage:
            self.coverage_layer = nn.Linear(1, hidden_dim, bias=False)

    def forward(self, value, query, padding_mask, pre_coverage):
        # value (batch_size, seq_len, hidden_dim*2)
        # query (batch_size, hidden_dim*2)
        # padding_mask (batch_size, seq_len)
        # pre_coverage (batch_size, seq_len)
        value = self.value_layer(value)                 # value (batch_size, seq_len, hidden_dim)
        query = self.query_layer(query.unsqueeze(1))    # query (batch_size, 1, hidden_dim)
        att_features = query+value
        if pre_coverage is not None:
            coverage = self.coverage_layer(pre_coverage.unsqueeze(2))
            # coverage (batch_size, seq_len, hidden_dim)
            att_features = att_features + coverage

        scores = self.energy_layer(torch.tanh((att_features))) # score (batch_size, seq_len, 1)
        scores = scores.squeeze(2)                             # score (batch_size, seq_len)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(padding_mask, -float('inf'))
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1) # alphas (batch_size, seq_len)
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas.unsqueeze(1), value)
        #context = (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_dim) = (batch_size, 1, hidden_dim)

        if pre_coverage is not None:
            pre_coverage = pre_coverage + alphas

        return context, alphas, pre_coverage

class Decoder(nn.Module):
    def __init__(self, embedding_matrix, padding_idx, hidden_dim, n_layers, dropout,
                 use_coverage, use_pointer_gen, cn_vocab_size=None, emb_dim=None):
        super().__init__()
        self.use_pointer_gen = use_pointer_gen
        if embedding_matrix is not None:
            self.emb_dim = embedding_matrix.shape[1]
            self.output_size = embedding_matrix.shape[0]
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=padding_idx)
        else:
            self.emb_dim = emb_dim
            self.output_size = cn_vocab_size
            self.embedding = nn.Embedding(cn_vocab_size,emb_dim)

        self.gru = nn.GRU(input_size=self.emb_dim + hidden_dim,
                          hidden_size=hidden_dim * 2,
                          num_layers=n_layers,
                          batch_first=True)
        self.attention = BahdanauAttention(hidden_dim=hidden_dim, use_coverage=use_coverage)
        self.output1 = nn.Linear(hidden_dim*2, hidden_dim * 4)
        self.output2 =nn.Linear(hidden_dim*4, hidden_dim * 8)
        self.output3 = nn.Linear(hidden_dim * 8, self.output_size)
        if self.use_pointer_gen:
            self.p_gen_linear = nn.Linear(hidden_dim * 4 + emb_dim, 1)

    def forward(self, inputs, enc_outputs, dec_hidden, padding_mask, pre_coverage):
        # input       (batch_size, 1)
        # enc_outputs (batch_size, length, hidden_dim*2)
        # dec_hidden  (num_layers, batch size, hid dim*2)
        context_vector, attention_weights, coverage = self.attention(enc_outputs, dec_hidden[-1,:,:], #用最后一层的dec_hidden去做attention
                                                                     padding_mask, pre_coverage)
        # context_vector(batch_size, 1, hidden_dim)
        x = self.embedding(inputs)                    # x (batch_size, 1, embedding_dim)
        x = torch.cat([context_vector, x], dim=2)     # x (batch_size, 1, embedding_dim+hidden_dim)
        outputs, hiddens = self.gru(x, dec_hidden)
        # outputs(batch_size, 1, 1*hidden_dim*2)
        # hiddens(directions*num_layers, batch_size, hidden_dim*2) 最后时刻的输出
        # outputs和hiddens的最后一层相等，因为这里只做了一个时刻

        prediction = self.output1(outputs.squeeze(1))
        prediction = self.output2(prediction)
        prediction = self.output3(prediction)        # prediction (batch_size, vocab_size)

        if self.use_pointer_gen:
            p_gen_input = torch.cat((context_vector.squeeze(1), outputs.squeeze(1), x.squeeze(1)), 1)
            # (batch_size, hidden_dim+hidden_dim+2*hidden_dim+emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

            vocab_dist = p_gen * F.softmax(prediction)
            attn_dist = (1 - p_gen) * attention_weights

        return prediction, hiddens, attention_weights, coverage