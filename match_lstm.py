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
        self.hidden_size = args.hidden_size
        h = self.hidden_size
        self.answer_token_len = args.answer_token_len
        
        self.embd = WordEmbedding(args)
        self.ctx_rnn   = nn.GRU(d, h, dropout = 0.2)
        self.query_rnn = nn.GRU(d, h, dropout = 0.2)
        
        self.ptr_net = PointerNetwork(h, h, self.answer_token_len) # TBD

        self.w  = nn.Parameter(torch.rand(1, h, 1).type(torch.FloatTensor), requires_grad=True) # (1, 1, h)
        self.Wq = nn.Parameter(torch.rand(1, h, h).type(torch.FloatTensor), requires_grad=True) # (1, h, h)
        self.Wp = nn.Parameter(torch.rand(1, h, h).type(torch.FloatTensor), requires_grad=True) # (1, h, h)
        self.Wr = nn.Parameter(torch.rand(1, h, h).type(torch.FloatTensor), requires_grad=True) # (1, h, h)
        # TODO bias

        # self.match_lstm_cell = nn.LSTMCell(2*h, h)
        self.match_lstm_cell = nn.GRUCell(2*h, h)

    def forward(self, context, query):
        # params
        d = self.embd_size
        h = self.hidden_size
        bs = context.size(0) # batch size
        T = context.size(1)  # context length 
        J = query.size(1)    # query length

        # LSTM Preprocessing Layer
        shape = (bs, T, J, h)
        embd_context     = self.embd(context)         # (N, T, d)
        embd_context, _h = self.ctx_rnn(embd_context) # (N, T, h)
        embd_context_ex  = embd_context.unsqueeze(2).expand(shape).contiguous() # (N, T, J, h)
        embd_query       = self.embd(query)           # (N, J, h)
        embd_query, _h   = self.query_rnn(embd_query) # (N, J, h)
        embd_query_ex    = embd_query.unsqueeze(1).expand(shape).contiguous() # (N, T, J, h)

        # Match-LSTM layer
        G = to_var(torch.zeros(bs, T, J, h)) # (N, T, J, h)
        
        wh_q = torch.bmm(embd_query, self.Wq.expand(bs, h, h)) # (N, J, h) = (N, J, h)(N, h, h)

        hidden     = to_var(torch.zeros([bs, h])) # (N, h)
        # cell_state = to_var(torch.zeros([bs, h])) # (N, h)
        # TODO bidirectional
        H_r = [hidden]
        # H_r = [hidden for _ in range(T)] # dummy
        for i in range(T):
            wh_p_i = torch.bmm(embd_context[:,i,:].clone().unsqueeze(1), self.Wp.expand(bs, h, h)) # (N, 1, h)
            wh_r_i = torch.bmm(hidden.unsqueeze(1), self.Wr.expand(bs, h, h)) # (N, 1, h)
            sec_elm = (wh_p_i + wh_r_i).expand(bs, J, h) # (N, J, h)

            G[:,i,:,:] = F.tanh( (wh_q + sec_elm).view(-1, h) ).view(bs, J, h) # (N, J, h) # TODO bias

            attn_i = torch.bmm(G[:,i,:,:].clone(), self.w.expand(bs, h, 1)).squeeze() # (N, J)
            attn_query = torch.bmm(attn_i.unsqueeze(1), embd_query).squeeze() # (N, h) = (N, 1, J)(N, J, H)
            z = torch.cat((embd_context[:,i,:], attn_query), 1) # (N, 2h)

            # hidden, cell_state = self.match_lstm_cell(z, (hidden, cell_state)) # (N, h), (N, h)
            hidden  = self.match_lstm_cell(z, hidden) # (N, h)
            H_r.append(hidden)
        H_r = torch.stack(H_r, dim=1) # (N, T, h)

        indices = self.ptr_net(H_r) # (N, M, T) , M means (start, end)
        return indices

