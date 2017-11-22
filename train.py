import os
import shutil
import pickle
import argparse
import time
import random
from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from process_data import save_pickle, load_pickle, load_task, load_processed_data,  load_glove_weights, to_var, make_word_vector
from jnet import JNet
from simple_net import SimpleNet
from match_lstm import MatchLSTM

def save_checkpoint(state, is_best, filename='checkpoint.tar', best_filename='model_best.tar'):
    print('save_model', filename, best_filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--embd_size', type=int, default=100, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=150, help='word embedding size')
parser.add_argument('--use_pickles', type=int, default=1, help='use pickles for dataset')
parser.add_argument('--resume', default='./model_best.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()

train_data = load_processed_data('./dataset/train.txt')
# dev_data = load_processed_data('./dataset/dev.txt')
data = train_data
# data = dev_data
vocab = set()
for ctx, query, _ in data:
    vocab |= set(ctx + query)
    
vocab = ['<PAD>', '<UNK'] + list(sorted(vocab))
w2i = dict((w, i) for i, w in enumerate(vocab, 0))
i2w = dict((i, w) for i, w in enumerate(vocab, 0))
args.vocab_size = len(vocab)
print('vocab size', len(vocab))

ctx_token_maxlen = max([len(c) for c, _, _ in data])
query_token_maxlen = max([len(q) for _, q, _ in data])
print('ctx_token_maxlen:', ctx_token_maxlen)
print('query_token_maxlen:', query_token_maxlen)
args.answer_token_len = 1 # 2 TODO


args.pre_embd = load_pickle('./pickles/glove_embd.pickle')
# args.pre_embd = torch.from_numpy(load_glove_weights('./dataset', args.embd_size, len(vocab), w2i)).type(torch.FloatTensor)
# save_pickle(args.pre_embd, './pickles/glove_embd.pickle')

def train(data, model, optimizer, loss_fn, n_epoch=5, batch_size=32):
    for epoch in range(n_epoch):
        print('---Epoch', epoch)
        random.shuffle(data)
        # start = time.time()
        for i in tqdm(range(0, len(data)-batch_size, batch_size)): # TODO shuffle, last elms
            # elapsed_time = time.time() - start
            # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            # start = time.time()

            batch_data = data[i:i+batch_size]
            c = [d[0] for d in batch_data]
            q = [d[1] for d in batch_data]
            b_ctx_token_maxlen = max([len(cc) for cc in c])
            b_query_token_maxlen = max([len(qq) for qq in q])
            print('BATCH context maxlen: {}, query maxlen: {}'.format(b_ctx_token_maxlen, b_query_token_maxlen))
            context_var = make_word_vector(c, w2i, b_ctx_token_maxlen)
            query_var   = make_word_vector(q, w2i, b_query_token_maxlen)
            labels = [d[2][0] for d in batch_data]
            labels = to_var(torch.LongTensor(labels))
            outs = model(context_var, query_var) # (B, M, L)
            # outs = outs.view(ctx_token_maxlen, -1).t().contiguous() # (B*M, L)
            outs = outs.squeeze()
            # outs = outs.view(-1, ctx_token_maxlen).contiguous() #(B*M, L)
            # outs = outs.view(-1, ctx_token_maxlen) #(B*M, L)
#             print('preds', torch.max(outs, 1)[1])
            labels = labels.view(-1) # (B*M)
            loss = loss_fn(outs, labels)
            if i % (batch_size*10) == 0:
                # print('outs[0]', outs[0][:100])
                print(loss.data)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        _, preds = torch.max(outs, 1)
        ct = 0
        for pred, label in zip(preds, labels):
            if pred.data[0] == label.data[0]: 
                ct+=1
        print('Acc', ct, '/', len(preds))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=True)
    
# model = JNet(args)
model = MatchLSTM(args)
# model = SimpleNet(args)
optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))#, lr=0.01)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if torch.cuda.is_available():
    model.cuda()

# print(model)
# for p in model.parameters():
#     print(p)
loss_fn = nn.NLLLoss()
# loss_fn = nn.CrossEntropyLoss()
train(train_data, model, optimizer, loss_fn)
print('fin')
