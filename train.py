import os
import pickle
import argparse
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from process_data import save_pickle, load_pickle, load_processed_data, load_glove_weights, to_var, make_word_vector
from utils import now, save_checkpoint, debug_log
# from jnet import JNet
# from simple_net import SimpleNet
from layers.match_lstm import MatchLSTM
from layers.attention_net import AttentionNet
from plotter import plot_heat_matrix


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
                    default=30,
                    help='number of epochs')
parser.add_argument('--test',
                    type=int,
                    default=0,
                    help='only run test() if 1')
parser.add_argument('--resume',
                    type=str,
                    # default='model_best.model',
                    default='',
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
# save_pickle(train_data, 'train_data.pickle')
# train_data = load_pickle('train_data.pickle')
dev_data = load_processed_data('./dataset/dev.txt')
# save_pickle(dev_data, 'dev_data.pickle')
# dev_data = load_pickle('dev_data.pickle')
print('train_data', len(train_data))
print('dev_data', len(dev_data))
data = train_data + dev_data
# vocab = set()
# for _, ctx, query, _, _ in data:
#     vocab |= set(ctx + query)
# vocab = ['<PAD>', '<UNK>'] + list(sorted(vocab))
# save_pickle(vocab, 'vocab.pickle')
vocab = load_pickle('vocab.pickle')
w2i = dict((w, i) for i, w in enumerate(vocab, 0))
i2w = dict((i, w) for i, w in enumerate(vocab, 0))
args.vocab_size = len(vocab)
print('vocab size', len(vocab))

ctx_token_maxlen = max([len(c) for _, c, _, _, _ in data])
query_token_maxlen = max([len(q) for _, _, q, _, _ in data])
print('ctx_token_maxlen:', ctx_token_maxlen)
print('query_token_maxlen:', query_token_maxlen)
args.answer_token_len = 1 # 2 TODO


# args.pre_embd = load_pickle('./pickles/glove_embd.pickle')
args.pre_embd = torch.from_numpy(load_glove_weights('./dataset', args.embd_size, len(vocab), w2i)).type(torch.FloatTensor)
# save_pickle(args.pre_embd, './pickles/glove_embd.pickle')


def train(model, data, test_data, optimizer, loss_fn, n_epoch=5, start_epoch=0, batch_size=32):
    debug_log('Training starts from {} to {}'.format(start_epoch, 'to', n_epoch))
    losses = {}
    for epoch in range(start_epoch, n_epoch+1):
        model.train()
        debug_log('---Epoch {}'.format(epoch))
        losses[str(epoch)] = []
        random.shuffle(data)
        for i in tqdm(range(0, len(data)-batch_size, batch_size)): # TODO use last elms
            batch_data = data[i:i+batch_size]
            c = [d[1] for d in batch_data]
            q = [d[2] for d in batch_data]
            # a_txt = [d[4] for d in batch_data]
            b_ctx_token_maxlen = max([len(cc) for cc in c])
            b_query_token_maxlen = max([len(qq) for qq in q])
            # print('BATCH context maxlen: {}, query maxlen: {}'.format(b_ctx_token_maxlen, b_query_token_maxlen))
            context_var = make_word_vector(c, w2i, b_ctx_token_maxlen)
            query_var   = make_word_vector(q, w2i, b_query_token_maxlen)
            labels = [d[3][0] for d in batch_data]
            labels = to_var(torch.LongTensor(labels))
            outs, attens = model(context_var, query_var) # (B, M, L), (B, L, J)

            # outs = outs.view(ctx_token_maxlen, -1).t().contiguous() # (B*M, L)
            outs = outs.squeeze()
            # outs = outs.view(-1, ctx_token_maxlen).contiguous() #(B*M, L)
            # outs = outs.view(-1, ctx_token_maxlen) #(B*M, L)
            labels = labels.view(-1) # (B*M)
            loss = loss_fn(outs, labels)
            if i % (batch_size*10) == 0:
                print('Epoch', epoch, ', Loss:', loss.data[0])
                losses[str(epoch)].append(loss.data[0])

                _, preds = torch.max(outs, 1)
                correct = torch.sum(preds == labels).data[0]
                print('correct: {:.2f}% ({}/{})'.format(correct/batch_size * 100, correct, batch_size))
                # for j, (pred, label) in enumerate(zip(preds, labels)):
                #     c_label = batch_data[j][0]
                #     if pred.data[0] == label.data[0]:
                #         save_fig_file = '{}/{}_TRAIN_{}_bs-{}_correct.png'.format(args.output_dir, now(), c_label, i)
                #     else:
                #         save_fig_file = '{}/{}_TRAIN_{}_bs-{}_wrong.png'.format(args.output_dir, now(), c_label, i)
                #     ans = batch_data[0][3]
                #     plot_heat_matrix(c[0], q[0], attens[0], ans, output_file=save_fig_file, title='Answer: '+a_txt[0], pred=pred.data[0])
                #     break # just one sample

            model.zero_grad()
            loss.backward()
            optimizer.step()

        # end epoch
        # _, preds = torch.max(outs, 1)
        # ct = 0
        # for pred, label in zip(preds, labels):
        #     if pred.data[0] == label.data[0]:
        #         ct += 1
        # debug_log('Current Acc: {:.2f}% ({}/{})'.format(ct/len(preds), ct, len(preds)))

        filename = '{}/{}_Epoch-{}.model'.format(args.output_dir, now(), epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=True, filename=filename)
        test(model, test_data)

    save_pickle(losses, '{}/{}_train_losses.pickle'.format(args.output_dir, now()))


# def test {{{
def test(model, data, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    debug_log('Test starts')
    for i in tqdm(range(0, len(data)-batch_size, batch_size)):  # TODO last elms
        batch_data = data[i:i+batch_size]
        c = [d[1] for d in batch_data]
        q = [d[2] for d in batch_data]
        # a_txt = [d[4] for d in batch_data]
        b_ctx_token_maxlen = max([len(cc) for cc in c])
        b_query_token_maxlen = max([len(qq) for qq in q])
        context_var = make_word_vector(c, w2i, b_ctx_token_maxlen)
        query_var   = make_word_vector(q, w2i, b_query_token_maxlen)
        labels = [d[3][0] for d in batch_data]
        labels = to_var(torch.LongTensor(labels))
        outs, attens = model(context_var, query_var) # (B, M, L), (B, L, J)
        outs = outs.squeeze()
        labels = labels.view(-1) # (B*M)
        loss = loss_fn(outs, labels)
        if i % (batch_size*10) == 0:
            print(loss.data[0])

        _, preds = torch.max(outs, 1)
        correct += torch.sum(preds == labels).data[0]
        # print(correct, '/', batch_size)
        # already_saved = False
        # for j, (pred, label) in enumerate(zip(preds, labels)):
        #     c_label = batch_data[j][0]
        #     if pred.data[0] == label.data[0]:
        #         correct += 1
        #         save_fig_file = '{}/{}_TEST_{}_bs-{}_correct{}.png'.format(args.output_dir, now(), c_label, i, j)
        #     else:
        #         save_fig_file = '{}/{}_TEST_{}_bs-{}_wrong{}.png'.format(args.output_dir, now(), c_label, i, j)
        #     ans = batch_data[j][3]
        #     if not already_saved:
        #         # test_img_data = (c[j], q[j], attens[j], ans, save_fig_file)
        #         # save_pickle(test_img_data, 'test_img_data.pickle')
        #         plot_heat_matrix(c[j], q[j], attens[j], ans, output_file=save_fig_file, title='Answer: '+a_txt[j], pred=pred.data[0])
        #     already_saved = True
        total += batch_size
    print('Test Acc: {:.2f}% ({}/{})'.format(100*correct/total, correct, total))
# }}}

# model = JNet(args)
# model = SimpleNet(args)
# model = MatchLSTM(args)
model = AttentionNet(args)
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
    train(model, train_data, dev_data, optimizer, loss_fn, args.n_epoch, args.start_epoch)

test(model, dev_data)

debug_log('Finish')
