import torch.nn as nn


# In : (N, sentence_len)
# Out: (N, sentence_len, embd_size)
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embd_size, pre_embd=None, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        if pre_embd is not None:
            print('Set pretrained embedding weights')
            self.embedding.weight = nn.Parameter(pre_embd, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)
