import os
import shutil
import pickle
import argparse
import time
from datetime import datetime
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from process_data import save_pickle, load_pickle, load_processed_data, load_glove_weights, to_var, make_word_vector
# from jnet import JNet
# from simple_net import SimpleNet
from match_lstm import MatchLSTM
from plotter import plot_heat_matrix


def save_checkpoint(state, is_best, filename='checkpoint.model', best_filename='model_best.model'):
    print('save_model', filename, best_filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def now():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def debug_log(text=''):
    msg = now()
    if text:
        msg += ': ' + text
    print(msg)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help='input batch size')
parser.add_argument('--embd_size',
                    type=int,
                    default=100,
                    help='word embedding size')
parser.add_argument('--hidden_size',
                    type=int,
                    default=150,
                    help='word embedding size')
parser.add_argument('--use_pickles',
                    type=int,
                    default=1,
                    help='use pickles for dataset')
parser.add_argument('--start_epoch',
                    type=int,
                    default=1,
                    help='initial epoch count')
parser.add_argument('--n_epoch',
                    type=int,
                    default=10,
                    help='number of epochs')
parser.add_argument('--test',
                    type=int,
                    default=0,
                    help='only run test() if 1')
parser.add_argument('--resume',
                    type=str,
                    default='best.model',
                    help='path to latest checkpoint')
parser.add_argument('--output_dir',
                    type=str,
                    default='./outputs',
                    metavar='PATH',
                    help='path to latest checkpoint')
args = parser.parse_args()
for arg in vars(args):
    print('Argument:', arg, getattr(args, arg))

train_data = load_processed_data('./dataset/train.txt')
dev_data = load_processed_data('./dataset/dev.txt')
data = train_data + dev_data
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


# args.pre_embd = load_pickle('./pickles/glove_embd.pickle')
args.pre_embd = torch.from_numpy(load_glove_weights('./dataset', args.embd_size, len(vocab), w2i)).type(torch.FloatTensor)
save_pickle(args.pre_embd, './pickles/glove_embd.pickle')


def train(data, model, optimizer, loss_fn, n_epoch=5, start_epoch=0, batch_size=32):
    debug_log('Training starts from {} to {}'.format(start_epoch, 'to', n_epoch))
    losses = {}
    for epoch in range(start_epoch, n_epoch+1):
        debug_log('---Epoch {}'.format(epoch))
        losses[str(epoch)] = []
        random.shuffle(data)
        # start = time.time()
        for i in tqdm(range(0, len(data)-batch_size, batch_size)): # TODO use last elms
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
            outs, attens = model(context_var, query_var) # (B, M, L), (B, L, J)

            # outs = outs.view(ctx_token_maxlen, -1).t().contiguous() # (B*M, L)
            outs = outs.squeeze()
            # outs = outs.view(-1, ctx_token_maxlen).contiguous() #(B*M, L)
            # outs = outs.view(-1, ctx_token_maxlen) #(B*M, L)
            labels = labels.view(-1) # (B*M)
            loss = loss_fn(outs, labels)
            if i % (batch_size*5) == 0:
                print('Epoch', epoch, ', Loss:', loss.data[0])
                losses[str(epoch)].append(loss.data[0])
                save_fig_file = '{}/{}_output_bs-{}_epoch-{}.png'.format(args.output_dir, now(), i, epoch)
                ans = batch_data[0][2]
                plot_heat_matrix(c[0], q[0], attens[0], ans, output_file=save_fig_file)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        # end epoch
        _, preds = torch.max(outs, 1)
        ct = 0
        for pred, label in zip(preds, labels):
            if pred.data[0] == label.data[0]:
                ct += 1
        debug_log('Current Acc: {:.2f}% ({}/{})'.format(ct/len(preds), ct, len(preds)))

        filename = '{}/{}_Epoch-{}.model'.format(args.output_dir, now(), epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=True, filename=filename)

    save_pickle(losses, '{}/{}_train_losses.pickle'.format(args.output_dir, now()))


def test(data, model, batch_size=32):
    correct = 0
    total = 0
    debug_log('Test starts')
    for i in tqdm(range(0, len(data)-batch_size, batch_size)):  # TODO last elms
        batch_data = data[i:i+batch_size]
        c = [d[0] for d in batch_data]
        q = [d[1] for d in batch_data]
        b_ctx_token_maxlen = max([len(cc) for cc in c])
        b_query_token_maxlen = max([len(qq) for qq in q])
        context_var = make_word_vector(c, w2i, b_ctx_token_maxlen)
        query_var   = make_word_vector(q, w2i, b_query_token_maxlen)
        labels = [d[2][0] for d in batch_data]
        labels = to_var(torch.LongTensor(labels))
        outs, attens = model(context_var, query_var) # (B, M, L), (B, L, J)
        outs = outs.squeeze()
        labels = labels.view(-1) # (B*M)
        loss = loss_fn(outs, labels)
        if i % (batch_size*10) == 0:
            # print('outs[0]', outs[0][:100])
            print(loss.data[0])
        model.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outs, 1)
        for pred, label in zip(preds, labels):
            if pred.data[0] == label.data[0]:
                correct += 1
        total += batch_size
    print('Test Acc: {:.2f}% ({}/{})'.format(correct/total, correct, total))

# model = JNet(args)
model = MatchLSTM(args)
# model = SimpleNet(args)
optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
loss_fn = nn.NLLLoss()

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

if args.test != 1:
    train(train_data, model, optimizer, loss_fn, args.n_epoch, args.start_epoch)

# test(dev_data, model)

debug_log('Finish')
