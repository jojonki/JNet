import torch
import torch.nn as nn
import torch.nn.functional as F

from word_embedding import WordEmbedding
from pointer_network import PointerNetwork

class JNet(nn.Module):
    def __init__(self, args):
        super(JNet, self).__init__()
        self.embd_size = args.embd_size
        h_emb = (int)(self.embd_size*0.5)
        self.answer_seq_len = args.answer_seq_len
        
        self.embd = WordEmbedding(args)
        self.ctx_birnn = nn.ModuleList([nn.GRU(self.embd_size, h_emb, bidirectional=True, dropout=0.2) for _ in range(2)])
        self.query_birnn = nn.ModuleList([nn.GRU(self.embd_size, h_emb, bidirectional=True, dropout=0.2) for _ in range(2)])
        self.last_rnn = nn.GRU(self.embd_size, self.embd_size, bidirectional=True, dropout=0.2)
#         self.last_layer = nn.Linear(args.ctx_sent_maxlen, args.ctx_sent_maxlen)
        
        self.ptr_net = PointerNetwork(self.embd_size*2, self.embd_size*2, args.ctx_sent_maxlen, self.answer_seq_len) # TBD
        
    def forward(self, context, query):
        N = context.size(0)
        JX = context.size(1)
        
        x = self.embd(context) # (N, JX, d)
        q_embd = self.embd(query) # (N, JQ, d)
        q, _h = self.ctx_birnn[0](q_embd) # (N, JQ, d)
        q = q.sum(1).unsqueeze(2) # (N, d, 1)

        for i in range(2):
            d, _h = self.ctx_birnn[i](x) # (N, JX, d)
        
            attn = F.softmax(torch.bmm(d, q).squeeze()) # (N, JX)
            attn_q = torch.bmm(attn.unsqueeze(2), q.transpose(2, 1)) # (N, JX, d)
            x = d * attn_q # (N, JX, d)
        x, _ = self.last_rnn(x) # (N, JX, 2d)
        indices = self.ptr_net(x) # (N, M, JX) , M means (start, end)
        indices = F.log_softmax(indices.view(-1, JX)).view(N, -1, JX) # (N, M, JX)
        return indices