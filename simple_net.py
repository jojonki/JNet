import torch
import torch.nn as nn
import torch.nn.functional as F

from word_embedding import WordEmbedding

class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.embd_size = args.embd_size

        self.embd           = WordEmbedding(args)
        self.W              = nn.Parameter(torch.rand(1, 3*self.embd_size, 1).type(torch.FloatTensor), requires_grad = True) # (N, 3d, 1) for bmm (N, T*J, 3d)
        self.modeling_layer = nn.GRU(3*self.embd_size, self.embd_size, bidirectional                                 = True)
        self.p1_layer       = nn.Linear(5*self.embd_size, args.ctx_token_maxlen)

    def forward(self, context, query):
        bs = context.size(0)
        T = context.size(1)
        J = query.size(1)
        embd_context = self.embd(context) # (N, T, d)
        embd_query = self.embd(query) # (N, J, d)

        shape = (bs, T, J, self.embd_size) # (N, T, J, d)
        embd_context_ex = embd_context.unsqueeze(2) # (N,T, 1, d)
        embd_context_ex = embd_context_ex.expand(shape) # (N, T, J, d)
        embd_query_ex = embd_query.unsqueeze(1) # (N, 1, J, d)
        embd_query_ex = embd_query_ex.expand(shape) # (N, T, J, d)
        combi = torch.mul(embd_context_ex, embd_query_ex)
        cat_data = torch.cat((embd_context_ex, embd_context_ex, combi), 3) # (N, T, J, 3d)
        cat_data = cat_data.view(bs, -1, cat_data.size(3)) # (N, T*J, 3d)
        S = torch.bmm(cat_data, self.W.expand(bs, self.W.size(1), 1)) # (N, T*J, 1)
        # S = S.view(bs, T, J)
        S = S.view(bs*T, J)
        S = torch.stack([F.softmax(S[i]) for i in range(len(S))], 0) # softmax for each row
        S = S.view(bs, T, J) # (N, T, J)

        c2q = torch.bmm(S, embd_query) # (N, T, d)
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q)), 2) # (N, T, 3d)
        M, _h = self.modeling_layer(G) # (M, T, 2d)
        G_M = torch.cat((G, M), 2) # (N, T, 5d)
        G_M = G_M.sum(1) # (N, 5d)
        p1 = F.log_softmax(self.p1_layer(G_M))

        return p1

