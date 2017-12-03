import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from process_data import load_pickle, load_task, load_glove_weights
# from process_data import to_var, make_word_vector, make_char_vector
from layers.word_embedding import WordEmbedding
from layers.pointer_network import PointerNetwork


class AttentionNet(nn.Module):
    '''
    BiDAF like networks
    https://arxiv.org/pdf/1611.01603.pdf
    '''
    def __init__(self, args):
        super(AttentionNet, self).__init__()
        self.embd_size = args.embd_size
        self.d = self.embd_size * 2

        self.word_embd_net = WordEmbedding(args.vocab_size, args.embd_size, args.pre_embd)
        self.ctx_embd_layer = nn.GRU(self.embd_size, self.embd_size, bidirectional=True, dropout=0.2)

        init_val = 0.01
        self.W = nn.Parameter(torch.Tensor(1, 3*self.d, 1).uniform_(-init_val, init_val).type(torch.FloatTensor), requires_grad=True) # (N, dd, 1)

        self.modeling_layer = nn.GRU(4*self.d, self.d, bidirectional=True, dropout=0.2)

        self.ptr_net = PointerNetwork(2*self.d, 2*self.d, answer_seq_len=1) # TBD

    def init_weights(self):
        # initialize weights if needed
        pass

    def build_contextual_embd(self, x):
        embd = self.word_embd_net(x) # (N, seq_len, embd_size)

        # 3. Contextual  Embedding Layer
        ctx_embd_out, _h = self.ctx_embd_layer(embd)
        return ctx_embd_out

    def forward(self, context, query):
        batch_size = context.size(0)
        T = context.size(1)   # context sentence length (word level)
        J = query.size(1) # query sentence length   (word level)

        # 1. Caracter Embedding Layer TODO
        # 2. Word Embedding Layer
        # 3. Contextual  Embedding Layer
        embd_context = self.build_contextual_embd(context) # (N, T, d)
        embd_query   = self.build_contextual_embd(query) # (N, J, d)

        # 4. Attention Flow Layer
        # Make a similarity matrix
        shape = (batch_size, T, J, self.d)            # (N, T, J, d)
        embd_context_ex = embd_context.unsqueeze(2)     # (N, T, 1, d)
        embd_context_ex = embd_context_ex.expand(shape) # (N, T, J, d)
        embd_query_ex = embd_query.unsqueeze(1)         # (N, 1, J, d)
        embd_query_ex = embd_query_ex.expand(shape)     # (N, T, J, d)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex) # (N, T, J, d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3) # (N, T, J, 3d), [h;u;h◦u]
        cat_data = cat_data.view(batch_size, -1, 3*self.d) # (N, T*J, 3d)
        S = torch.bmm(cat_data, self.W.repeat(batch_size, 1, 1)).squeeze() # (N, T*J)
        S = S.view(batch_size*T, J) # (N*T, J)
        S = F.softmax(S)
        S = S.view(batch_size, T, J) # (N, T, J)

        # Context2Query
        c2q = torch.bmm(S, embd_query) # (N, T, d) = bmm( (N, T, J), (N, J, d) )
        c2q = F.softmax(c2q.view(batch_size*T, -1))
        c2q = c2q.view(batch_size, T, self.d) # (N, T, d)

        # Query2Context
        # b: attention weights on the context
        tmp_b = torch.max(S, 2)[0] # (N, T)
        b = torch.stack([F.softmax(tmp_b[i]) for i in range(batch_size)], 0) # (N, T), softmax for each row
        q2c = torch.bmm(b.unsqueeze(1), embd_context) # (N, 1, d) = bmm( (N, 1, T), (N, T, d) )
        q2c = q2c.repeat(1, T, 1) # (N, T, d), tiled T times

        # G: query aware representation of each context word
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2) # (N, T, 4d)

        # 5. Modeling Layer
        M, _h = self.modeling_layer(G) # M: (N, T, 2d)

        # out = self.ptr_net(c2q)
        out = self.ptr_net(M) # (N, T)

        return out, c2q
