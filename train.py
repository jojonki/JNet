import json
import pickle
import argparse
# from nltk.tokenize import word_tokenize
from tqdm import tqdm
from process_data import save_pickle, load_pickle, load_task, load_glove_weights, to_var, make_word_vector
# from word_embedding import WordEmbedding
import torch
# import torch.nn as nn
# from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from jnet import JNet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--embd_size', type=int, default=100, help='word embedding size')
parser.add_argument('--use_pickles', type=int, default=1, help='use pickles for dataset')
args = parser.parse_args()

# if args.use_pickles == 1:
#     train_data = load_pickle('./pickles/train_data.pickle')
#     dev_data = load_pickle('./pickles/dev_data.pickle')
# else:
#     train_data, train_ctx_maxlen = load_task('./dataset/train-v1.1.json')
#     save_pickle(train_data, './pickles/train_data.pickle')
#     dev_data, dev_ctx_maxlen = load_task('./dataset/dev-v1.1.json')
#     save_pickle(dev_data, './pickles/dev_data.pickle')

train_data = load_processed_data('./dataset/train.txt')
data = train_data
vocab = set()
for ctx, query, _ in data:
    vocab |= set(ctx + query)
    
vocab = list(sorted(vocab))
w2i = dict((w, i) for i, w in enumerate(vocab, 0))
i2w = dict((i, w) for i, w in enumerate(vocab, 0))
args.vocab_size = len(vocab)
print('vocab size', len(vocab))

ctx_token_maxlen = max([len(c) for c, _, _ in data])
query_token_maxlen = max([len(q) for _, q, _ in data])
print('ctx_token_maxlen:', ctx_token_maxlen)
print('query_token_maxlen:', query_token_maxlen)
args.ctx_token_maxlen = ctx_token_maxlen
args.query_token_maxlen = query_token_maxlen
args.answer_seq_len = 1 # 2 TODO

glove_embd = load_pickle('./pickles/glove_embd.pickle')
# glove_embd = torch.from_numpy(load_glove_weights('./dataset', args.embd_size, len(vocab), w2i)).type(torch.FloatTensor)
# save_pickle(glove_embd, './pickles/glove_embd.pickle')
args.pre_embd = glove_embd

def train(model, optimizer, loss_fn, n_epoch=5, batch_size=256):
    for epoch in range(n_epoch):
        print('---Epoch', epoch)
        for i in tqdm(range(0, len(data)-batch_size, batch_size)): # TODO shuffle, last elms
            batch_data = data[i:i+batch_size]
            c = [d[0] for d in batch_data]
            q = [d[2] for d in batch_data]
            context_var = make_word_vector(c, w2i, ctx_sent_maxlen)
            query_var = make_word_vector(q, w2i, query_sent_maxlen)
#             labels = [[d[4][0], d[5][0]] for d in batch_data]
            labels = [d[4][0] for d in batch_data]
            labels = to_var(torch.LongTensor(labels))
            outs = model(context_var, query_var)
            outs = outs.view(-1, ctx_sent_maxlen) #(B*M, L)
            labels = labels.view(-1) # (B*M)
            loss = loss_fn(outs, labels)
            if i % (batch_size*10) == 0:
                print(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             break
        
        _, preds = torch.max(outs, 1)
        ct = 0
        for pred, label in zip(preds, labels):
            if pred.data[0] == label.data[0]: 
                ct+=1
        print('Acc', ct, '/', len(preds))
        break
        
    
model = JNet(args)
if torch.cuda.is_available():
    model.cuda()

print(model)
for p in model.parameters():
    print(p)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
train(model, optimizer, loss_fn)
print('fin')
