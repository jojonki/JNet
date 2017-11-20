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
        self.embd_size = args.embd_size
        d = self.embd_size
        # h_d = (int)(0.5*d)
        self.answer_token_len = args.answer_token_len
        
        self.embd = WordEmbedding(args)
        self.ctx_rnn   = nn.GRU(d, d, dropout = 0.2)
        self.query_rnn = nn.GRU(d, d, dropout = 0.2)
        # self.last_rnn = nn.GRU(d, d, bidirectional=True, dropout=0.2)
        
        # self.ptr_net = PointerNetwork(d*2, d*2, args.ctx_token_maxlen, self.answer_token_len) # TBD
        self.ptr_net = PointerNetwork(d, d, self.answer_token_len) # TBD

        self.w  = nn.Parameter(torch.rand(1, d, 1).type(torch.FloatTensor), requires_grad=True) # (1, 1, d)
        self.Wq = nn.Parameter(torch.rand(1, d, d).type(torch.FloatTensor), requires_grad=True) # (1, d, d)
        self.Wp = nn.Parameter(torch.rand(1, d, d).type(torch.FloatTensor), requires_grad=True) # (1, d, d)
        self.Wr = nn.Parameter(torch.rand(1, d, d).type(torch.FloatTensor), requires_grad=True) # (1, d, d)
        # TODO bias

        self.match_lstm_cell = nn.LSTMCell(2*d, d)

    def forward(self, context, query):
        # params
        d = self.embd_size
        bs = context.size(0) # batch size
        T = context.size(1)  # context length 
        J = query.size(1)    # query length

        # LSTM Preprocessing Layer
        embd_context     = self.embd(context)         # (N, T, d)
        embd_context, _h = self.ctx_rnn(embd_context) # (N, T, d)
        embd_query       = self.embd(query)           # (N, J, d)
        embd_query, _h   = self.query_rnn(embd_query) # (N, J, d)

        # Match-LSTM layer
        # attention = to_var(torch.zeros(bs, T, J)) # (N, T, J)
        G = to_var(torch.zeros(bs, T, J, d)) # (N, T, J, d)

        wh_q = torch.bmm(embd_query, self.Wq.expand(bs, d, d)) # (N, J, d) = (N, J, d)(N, d, d)

        hidden     = to_var(torch.randn([bs, d])) # (N, d)
        cell_state = to_var(torch.randn([bs, d])) # (N, d)
        # TODO bidirectional
        H_r = [hidden]
        for i in range(T):
            wh_p_i = torch.bmm(embd_context[:,i,:].clone().unsqueeze(1), self.Wp.expand(bs, d, d)).squeeze() # (N, 1, d) -> (N, d)
            wh_r_i = torch.bmm(hidden.unsqueeze(1), self.Wr.expand(bs, d, d)).squeeze() # (N, 1, d) -> (N, d)
            sec_elm = (wh_p_i + wh_r_i).unsqueeze(1).expand(bs, J, d) # (N, J, d)

            G[:,i,:,:] = F.tanh( (wh_q + sec_elm).view(-1, d) ).view(bs, J, d) # (N, J, d) # TODO bias

            attn_i = torch.bmm(G[:,i,:,:].clone(), self.w.expand(bs, d, 1)).squeeze() # (N, J)
            attn_query = torch.bmm(attn_i.unsqueeze(1), embd_query).squeeze() # (N, d) 
            z = torch.cat((embd_context[:,i,:], attn_query), 1) # (N, 2d)

            hidden, cell_state = self.match_lstm_cell(z, (hidden, cell_state)) # (N, d), (N, d)
            H_r.append(hidden)
        H_r = torch.stack(H_r, dim=1) # (N, T, d)

        indices = self.ptr_net(H_r) # (N, M, T) , M means (start, end)
        return indices

