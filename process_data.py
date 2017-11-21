import os
import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize
import torch
from torch.autograd import Variable

def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)

def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)

def lower_list(str_list):
    return [str_var.lower() for str_var in str_list]

def load_processed_data(fpath):
    with open(fpath) as f:
        lines = f.readlines()
        data = []
        for l in lines:
            c, q, a = l.rstrip().split('\t')
            c, q, a = c.split(' '), q.split(' '), a.split(' ')
            # if len(c) > 30: continue # TMP
            c, q = lower_list(c), lower_list(q)
            a = [int(aa) for aa in a]
            a = [a[0], a[-1]]
            data.append((c, q, a))
    return data

def load_task(dataset_path):
    ret_data = []
    ctx_max_len = 0 # character level length
    with open(dataset_path) as f:
        data = json.load(f)
        ver = data['version']
        # print('dataset version:', ver)
        data = data['data']
        for i, d in enumerate(data):
            if i % 100 == 0: print('load_task:', i, '/', len(data))
            # print('load', d['title'], i, '/', len(data))
            for p in d['paragraphs']:
                if len(p['context']) > ctx_max_len:
                    ctx_max_len = len(p['context'])
                c = word_tokenize(p['context'])
                q, a = [], []
                for qa in p['qas']:
                    q = word_tokenize(qa['question'])
                    a = [ans['text'] for ans in qa['answers']]
                    # a_beg = [ans['answer_start'] for ans in qa['answers']]
                    for ans in qa['answers']:
                        try:
                            a_beg = [c.index(ans['text'].split(' ')[0]) for ans in qa['answers']]
                            # a_beg = c.index(a[0].split(' ')[0])
                            a_end = [ans['answer_start'] + len(ans['text']) for ans in qa['answers']]

                            ret_data.append((c, qa['id'], q, a, a_beg, a_end))
                        except ValueError:
                            pass
                            # print('Invalid QA')
                            # print('Context:', c)
                            # print('Querstion:', q)
                            # print('Answer', ans['text'].split(' ')[0])
                    # break
                # break
    return ret_data, ctx_max_len

def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.' + str(embd_dim) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index)) 
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    count = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            count += 1
        # else:
        #     print('not found', word)
        #     if count > 1000: break
    print('Use pre-embedded weights:', count, '/', len(word_index.items()))

    return embedding_matrix

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def add_padding(data, seq_len):
    pad_len = max(0, seq_len - len(data))
    data += [0] * pad_len
    data = data[:seq_len]
    return data

def make_word_vector(data, w2i_w, query_len):
    vec_data = []
    for sentence in data:
        index_vec = [w2i_w[w] for w in sentence]
        index_vec = add_padding(index_vec, query_len)
        vec_data.append(index_vec)
    
    return to_var(torch.LongTensor(vec_data))

def make_one_hot(data, ans_size):
    vec_data = []
    for ans_id in data:
        tmp = [0] * ans_size
        tmp[ans_id] = 1
        vec_data.append(tmp)
    return to_var(torch.LongTensor(vec_data))
