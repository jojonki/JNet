import torch
import torch.nn as nn
import torch.nn.functional as F

# In : (N, sentence_len)
# Out: (N, sentence_len, embd_size)
class WordEmbedding(nn.Module):
    def __init__(self, args, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embd_size)
        if args.pre_embd is not None:
            self.embedding.weight = nn.Parameter(args.pre_embd, requires_grad=is_train_embd)
        
    def forward(self, x):
        return F.relu(self.embedding(x))

