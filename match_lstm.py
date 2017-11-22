import torch
import torch.nn as nn
import torch.nn.functional as F

from word_embedding import WordEmbedding
from pointer_network import PointerNetwork
from process_data import to_var


class MatchLSTM(nn.Module):
    '''
        Machine Comprehension Using Match-LSTM and Answer Pointer
        Shuohang Wang, Jing Jiang
        https://arxiv.org/abs/1608.07905

        PointerNetwork works as the Boundary Model
    '''
    def __init__(self, args):
        super(MatchLSTM, self).__init__()
        initrange = 0.1
        self.initrange = initrange

        self.embd_size        = args.embd_size
        d                     = self.embd_size
        self.hidden_size      = args.hidden_size
        h                     = self.hidden_size
        self.answer_token_len = args.answer_token_len

        self.embd = WordEmbedding(args)
        self.ctx_rnn   = nn.GRU(d, h, dropout=0.2)
        self.query_rnn = nn.GRU(d, h, dropout=0.2)

        self.w  = nn.Parameter(torch.Tensor(1, h, 1).uniform_(-initrange, initrange), requires_grad=True) # (1, 1, h)
        self.Wq = nn.Parameter(torch.Tensor(1, h, h).uniform_(-initrange, initrange), requires_grad=True) # (1, h, h)
        self.Wp = nn.Parameter(torch.Tensor(1, h, h).uniform_(-initrange, initrange), requires_grad=True) # (1, h, h)
        self.Wr = nn.Parameter(torch.Tensor(1, h, h).uniform_(-initrange, initrange), requires_grad=True) # (1, h, h)
        self.gate_bias = nn.Parameter(torch.zeros(1, 1, h))
        self.attn_bias = nn.Parameter(torch.zeros(1, 1))

        # self.match_lstm_cell = nn.LSTMCell(2*h, h)
        self.match_lstm_cell   = nn.GRUCell(2*h, h)
        self.match_lstm_cell_r = nn.GRUCell(2*h, h)

        # PointerNetworks for boundary model
        self.ptr_net = PointerNetwork(2*h, 2*h, self.answer_token_len) # TBD

        self.init_weights()

    def init_weights(self):
        pass
        # initrange = self.initrange
        # self.ctx_rnn.weight.data.uniform_(-initrange, initrange)

    def forward(self, context, query):
        # params
        # d  = self.embd_size
        h  = self.hidden_size
        bs = context.size(0)  # batch size
        T  = context.size(1)  # context length
        J  = query.size(1)    # query length

        # LSTM Preprocessing Layer
        # shape = (bs, T, J, h)
        embd_context     = self.embd(context)         # (N, T, d)
        embd_context, _h = self.ctx_rnn(embd_context) # (N, T, h)
        # embd_context_ex  = embd_context.unsqueeze(2).expand(shape).contiguous() # (N, T, J, h)
        embd_query       = self.embd(query)           # (N, J, h)
        embd_query, _h   = self.query_rnn(embd_query) # (N, J, h)
        # embd_query_ex    = embd_query.unsqueeze(1).expand(shape).contiguous() # (N, T, J, h)

        # Match-LSTM layer
        G   = to_var(torch.zeros(bs, T, J, h)) # (N, T, J, h)
        G_r = to_var(torch.zeros(bs, T, J, h)) # (N, T, J, h)

        wh_q = torch.bmm(embd_query, self.Wq.expand(bs, h, h)) # (N, J, h) = (N, J, h)(N, h, h)

        h_r   = to_var(torch.zeros([bs, h])) # (N, h)
        h_r_r = to_var(torch.zeros([bs, h])) # (N, h)
        H_r   = [torch.cat((h_r, h_r_r), 1)] # [(N, 2h)]
        for i in range(T):
            j = T - 1 - i # reverse index

            wh_p_i = torch.bmm(embd_context[:, i, :].clone().unsqueeze(1), self.Wp.expand(bs, h, h)) # (N, 1, h)

            wh_r_i   = torch.bmm(h_r.unsqueeze(1),   self.Wr.expand(bs, h, h)) # (N, 1, h)
            wh_r_i_r = torch.bmm(h_r_r.unsqueeze(1), self.Wr.expand(bs, h, h)) # (N, 1, h)

            bias = self.gate_bias.expand(bs, J, h)

            sec_elm   = (wh_p_i + wh_r_i  ).expand(bs, J, h) # (N, J, h)
            sec_elm_r = (wh_p_i + wh_r_i_r).expand(bs, J, h) # (N, J, h)

            G[:, i, :, :]   = F.tanh( (wh_q + sec_elm   + bias).view(-1, h) ).view(bs, J, h) # (N, J, h)
            G_r[:, j, :, :] = F.tanh( (wh_q + sec_elm_r + bias).view(-1, h) ).view(bs, J, h) # (N, J, h)

            attn_bias = self.attn_bias.expand(bs, J)
            attn_i   = F.softmax(torch.bmm(G[:, i, :, :].clone(), self.w.expand(bs, h, 1)).squeeze() + attn_bias ) # (N, J)
            attn_i_r = F.softmax(torch.bmm(G_r[:, j, :, :].clone(), self.w.expand(bs, h, 1)).squeeze() + attn_bias) # (N, J)
            attn_query   = torch.bmm(attn_i.unsqueeze(1),   embd_query).squeeze() # (N, h) = (N, 1, J)(N, J, H)
            attn_query_r = torch.bmm(attn_i_r.unsqueeze(1), embd_query).squeeze() # (N, h) = (N, 1, J)(N, J, H)
            z   = torch.cat((embd_context[:, i, :], attn_query),   1) # (N, 2h)
            z_r = torch.cat((embd_context[:, j, :], attn_query_r), 1) # (N, 2h)

            h_r    = self.match_lstm_cell(z, h_r) # (N, h)
            h_r_r  = self.match_lstm_cell_r(z_r, h_r_r) # (N, h)
            H_r.append(torch.cat((h_r, h_r_r), 1)) # [(N, 2h)]
        H_r = torch.stack(H_r, dim=1) # (N, T, 2h)

        indices = self.ptr_net(H_r) # (N, M, T) , M means (start, end)
        return indices
