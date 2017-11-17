import torch
import torch.nn as nn
import torch.nn.functional as F
from process_data import to_var

class PointerNetwork(nn.Module):
    def __init__(self, hidden_size, weight_size, seq_len, answer_seq_len=2):
        super(PointerNetwork, self).__init__()

        self.hidden_size = hidden_size
        # self.input_size = input_size 
        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size 
        self.seq_len = seq_len
        # self.emb_size = emb_size 

        # self.emb = nn.Embedding(input_size, emb_size)  # embed inputs
        self.enc = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # TODO bidirectional
        self.dec = nn.LSTMCell(hidden_size, hidden_size)
        self.W1 = nn.Linear(hidden_size, weight_size) # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size) # blending decoder
        self.vt = nn.Linear(weight_size, 1) # scaling sum of enc and dec by v.T

        self.tanh = nn.Tanh()

    def forward(self, input):
        batch_size = input.size(0)
        # input = self.emb(input) # (N, L, hidden_size)
        # Encoding
        encoder_states, hc = self.enc(input) # encoder_state: (N, L, H)
        encoder_states = encoder_states.transpose(1, 0) # (L, N, H)

        # Decoding states initialization
        decoder_input = to_var(torch.Tensor(batch_size, self.hidden_size).zero_()) # (N, hidden_size)
        hidden = to_var(torch.randn([batch_size, self.hidden_size]))            # (N, h)
        cell_state = encoder_states[-1]                                         # (N, h)

        probs = []
        # Decoding
        for i in range(self.answer_seq_len): # range(M)
            hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (N, h), (N, h)

            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)        # (L, N, W)
            blend2 = self.W2(hidden)                # (N, W)
            blend_sum = self.tanh(blend1 + blend2)  # (L, N, W)
            out = self.vt(blend_sum)                # (L, N, 1)
            out = torch.squeeze(out)                # (L, N)
            probs.append(out)

        probs = torch.stack(probs, dim=2) # (L, N, M)
        probs = probs.permute(1, 2, 0)    #(N, M, L)

        return F.log_softmax(probs)
