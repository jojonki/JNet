import torch
import torch.nn as nn
import torch.nn.functional as F

from word_embedding import WordEmbedding
from pointer_network import PointerNetwork
from process_data import to_var

class JNet(nn.Module):
    def __init__(self, args):
        super(JNet, self).__init__()
        self.embd_size = args.embd_size
        h_emb = (int)(self.embd_size*0.5)
        # h_emb = self.embd_size
        self.answer_token_len = args.answer_token_len
        
        self.embd = WordEmbedding(args)
        self.ctx_birnn = nn.ModuleList([nn.GRU(self.embd_size, h_emb, bidirectional=True, dropout=0.2) for _ in range(2)])
        self.query_birnn = nn.ModuleList([nn.GRU(self.embd_size, h_emb, bidirectional=True, dropout=0.2) for _ in range(2)])
        self.last_rnn = nn.GRU(self.embd_size, self.embd_size, bidirectional=True, dropout=0.2)
        
        self.ptr_net = PointerNetwork(self.embd_size, self.embd_size, self.answer_token_len) # TBD

        self.w = nn.Parameter(torch.rand(1, self.embd_size).type(torch.FloatTensor), requires_grad = True) # (1, d)
        self.Wq = nn.Parameter(torch.rand(1, self.embd_size, self.embd_size).type(torch.FloatTensor), requires_grad = True) # (1, d, d)
        self.Wp = nn.Parameter(torch.rand(1, self.embd_size, self.embd_size).type(torch.FloatTensor), requires_grad = True) # (1, d, d)
        self.Wr = nn.Parameter(torch.rand(1, self.embd_size, self.embd_size).type(torch.FloatTensor), requires_grad = True) # (1, d, d)
        # TODO bias

        # self.match_lstm = nn.GRU(self.embd_size, self.embd_size, bidirectional=True, dropout=0.2)
        self.dec = nn.LSTMCell(2*self.embd_size, self.embd_size)

    def forward(self, context, query):
        bs = context.size(0) # batch size
        T = context.size(1)  # context length 
        J = query.size(1)    # query length

        # LSTM Preprocessing Layer
        embd_context = self.embd(context) # (N, T, d)
        embd_query = self.embd(query)     # (N, J, d)

        # Match-LSTM layer
        attention = to_var(torch.zeros(bs, T, J)) # (N, T, J)
        G = to_var(torch.zeros(bs, T, J, self.embd_size)) # (N, T, J, d)

        shape = (bs, T, J, self.embd_size) # (N, T, J, d)
        embd_query_ex = embd_query.unsqueeze(1).expand(shape) # (N, T, J, d)
        embd_context_ex = embd_context.unsqueeze(2).expand(shape) # (N, T, J, d)

        wh_q = torch.bmm(embd_query, self.Wq.expand(bs, self.embd_size, self.embd_size)) # (N, J, d)

        # wh_p = torch.bmm(embd_context, self.Wp.expand(bs, self.embd_size, self.embd_size)) # (N, T, d)

        decoder_input = to_var(torch.Tensor(bs, self.embd_size).zero_()) # (N, d)
        hidden = to_var(torch.randn([bs, self.embd_size])) # (N, d)
        cell_state = to_var(torch.randn([bs, self.embd_size])) # (N, d)
        # Decoding
        # TODO bidirectional
        H_r = [hidden]
        for i in range(T):
            wh_p_i = torch.bmm(embd_context[:,i,:].unsqueeze(1), self.Wp.expand(bs, self.embd_size, self.embd_size)).squeeze() # (N, 1, d) -> (N, d)
            wh_r_i = torch.bmm(hidden.unsqueeze(1), self.Wr.expand(bs, self.embd_size, self.embd_size)).squeeze() # (N, 1, d) -> (N, d)
            sec_elm = (wh_p_i + wh_r_i).unsqueeze(1).expand(bs, J, self.embd_size) # (N, J, d)

            G[:,i,:] = F.tanh( (wh_q + sec_elm).view(-1, self.embd_size) ).view(bs, J, self.embd_size) # (N, J, d) # TODO bias

            attn_i = attention[:,i,:] # (N, J)
            attn_query = torch.bmm(attn_i.unsqueeze(1), embd_query).squeeze() # (N, d) 
            z = torch.cat((embd_context[:,i,:], attn_query), 1) # (N, 2d)
            hidden, cell_state = self.dec(z, (hidden, cell_state)) # (N, d), (N, d)
            H_r.append(hidden)
        H_r = torch.stack(H_r, dim=1) # (N, T, d)

        indices = self.ptr_net(H_r) # (N, M, T) , M means (start, end)
        return indices


    def forward4(self, context, query):
        x = self.embd(context) # (N, JX, d)
        q_embd = self.embd(query) # (N, JQ, d)
        q, _h = self.query_birnn[0](q_embd) # (N, JQ, d)
        q = q.sum(1).unsqueeze(1) # (N, 1, d)
        x = x + q
        x = self.tmp_linear(x.sum(1))


        return x

    def forward3(self, context, query):
        N = context.size(0)  # Number of samples
        JX = context.size(1) # Number of tokens (word level context length)
        # print('context', context.size())
        # print('query', query.size())
        
        x = self.embd(context) # (N, JX, d)
        q_embd = self.embd(query) # (N, JQ, d)
        q, _h = self.query_birnn[0](q_embd) # (N, JQ, d)
        q = q.sum(1).unsqueeze(1) # (N, 1, d)

        x = x + q # (N, JX, d)
        # print('x', x[0])

        # x, _ = self.last_rnn(x) # (N, JX, 2d)
        indices = self.ptr_net(x) # (N, M, JX) , M means (start, end)
        # print(indices[0])
        # indices = F.log_softmax(indices.view(-1, JX)).view(N, -1, JX) # (N, M, JX)
        # print(indices[0][0])
        # indices = x.sum(2).unsqueeze(1)
        return indices
        
    def forward2(self, context, query):
        N = context.size(0)  # Number of samples
        JX = context.size(1) # Number of tokens (word level context length)
        
        x = self.embd(context) # (N, JX, d)
        q_embd = self.embd(query) # (N, JQ, d)
        q, _h = self.query_birnn[0](q_embd) # (N, JQ, d)
        q = q.sum(1).unsqueeze(2) # (N, d, 1)

        for i in range(2):
            d, _h = self.ctx_birnn[i](x) # (N, JX, d)
        
            attn = F.softmax(torch.bmm(d, q).squeeze()) # (N, JX)
            attn_q = torch.bmm(attn.unsqueeze(2), q.transpose(2, 1)) # (N, JX, d)
            x = d * attn_q # (N, JX, d)
        x, _ = self.last_rnn(x) # (N, JX, 2d)
        indices = self.ptr_net(x) # (N, M, JX) , M means (start, end)
        # indices = F.log_softmax(indices.view(-1, JX)).view(N, -1, JX) # (N, M, JX)
        return indices
