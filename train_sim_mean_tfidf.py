# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:13:56 2016

@author: hedi
"""

import numpy as np
import sys
from time import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sim_model import m_sim_mean
import theano
from tokenizer import word_tokenize
import json
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

fwrite = sys.stdout.write
dtype = theano.config.floatX

if __name__=='__main__':
    np.random.seed(1234)
    which_set = 'r'
    fwrite('Loading data...')
    data = np.load(which_set+'_similarity_data').all()
    fwrite(' Ok\n')
    pooling = 'mean' #mean, max or softmax
    optim = 'sgd' #sgd or rmsprop (but rmsprop is really slow...)
    folder = os.path.join(which_set+'_sim_tfidf', pooling)
    if not os.path.exists(folder):
        os.mkdir(folder)
    saving_path = os.path.join(folder, "model.mod")
    
    d = data[data.keys()[0]]
    dim_img = d['image_emb'].shape[0]
    dim_multi = 150
    
    image_embeddings = []
    product_ids = []
    fwrite('Loading data ...\n')
    product_ids = data.keys()
    tf = []
    df = Counter()
    for idx in product_ids:
        d = data.pop(idx)
        if not len(data)%5000:
            fwrite('%d\n' % len(data))
        image_embeddings.append(d['image_emb'].astype(dtype))
        
        tokenized = word_tokenize(json.loads(d['product'])['Description'])
        tfs = {}
        for w,c in Counter(tokenized).iteritems(): 
            tfs[w] = float(c) / len(tokenized)
        for w in set(tokenized):
            df[w] += 1
        tf.append(tfs)
    del data
    
    D = len(product_ids)
    idfs = dict((k, np.log(float(D)/df[k])) for k in df if df[k]>5)
    del df
    vocab = idfs.keys()
    vocab_dict = dict((k,v) for v,k in enumerate(vocab))
    vocab_size = len(vocab)
    dim_txt = vocab_size
    
    text_embeddings = []
    for freqs in tf:
        tfidf = np.zeros(vocab_size)
        for w in freqs:
            if w in idfs:
                tfidf[vocab_dict[w]] = freqs[w] * idfs[w]
        text_embeddings.append(tfidf)
    
    n_data = len(image_embeddings)
    fwrite('Done\n')
    product_ids = np.array(product_ids[:n_data])
    
    image_embeddings = np.array(image_embeddings[:n_data])
    text_embeddings = np.array(text_embeddings[:n_data])
    T = text_embeddings.sum(axis=1)**2
    zero_indexes = np.where(T==0.)[0]
    text_embeddings = np.delete(text_embeddings, zero_indexes, axis=0)
    image_embeddings = np.delete(image_embeddings, zero_indexes, axis=0)
    n_data = text_embeddings.shape[0]
    fwrite('Number of empty texts = %d\n' % len(zero_indexes))
    indexes = np.arange(n_data)
    np.random.shuffle(indexes)
    
    image_embeddings = image_embeddings[indexes]
    text_embeddings = text_embeddings[indexes]
    product_ids = product_ids[indexes]
    
    
    n_train = int(0.7*n_data)
    Xim_train = image_embeddings[:n_train]
    Xim_test = image_embeddings[n_train+1:]
    Xtxt_train = text_embeddings[:n_train]
    Xtxt_test = text_embeddings[n_train+1:]
    product_ids_train = product_ids[:n_train]
    product_ids_test = product_ids[n_train + 1 :]
    n_test = Xim_test.shape[0]
    lr_img = 0.001
    lr_txt = 0.001
    l2 = 0.
    K = 5
    batch_size = 16
    n_batches = n_train/batch_size
    fwrite('dim_img = %d, dim_txt = %d, dim_multi = %d\n' % (dim_img, dim_txt, dim_multi))
    model = m_sim_mean.create(dim_img, dim_txt, dim_multi)
    train_idxs = np.arange(n_train)

    for epoch in range(51):
        if not epoch % 5:
            test_rank = model.test(Xim_test, Xtxt_test, epoch=0, saving_path = folder)
            fwrite('\tTest median rank = %d\n' % test_rank)
            sys.stdout.flush()
        np.random.shuffle(train_idxs)
        Xim_train = Xim_train[train_idxs]
        Xtxt_train = Xtxt_train[train_idxs]
        fwrite('Epoch: %d, W_im = %f, W_txt = %f\n' % (epoch,(model.W_img**2).mean(), 
                                                       (model.W_txt**2).mean()))
        tic = time()
        model.train(Xim_train, Xtxt_train, K, lr_img, lr_txt, batch_size=batch_size, verbose=True)
        toc = time()-tic
        model.save(saving_path)
        fwrite('\tTime = %fs\n' % toc)