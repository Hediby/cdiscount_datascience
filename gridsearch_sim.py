# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 08:33:54 2016

@author: hedi
"""

import numpy as np
import sys
from time import time
import os
from sim_model import m_sim_mean
import theano
from itertools import product
import json
from joblib import Parallel, delayed

fwrite = sys.stdout.write
dtype = theano.config.floatX

def run_gristep(parameters, Xim_train, Xtxt_train):
    lr_img, lr_txt, K, batch_size = parameters
    model = m_sim_mean.create(dim_img, dim_txt, dim_multi)
    model_name = '_'.join(str(p).translate(None, '.') for p in parameters)
    f = open('gridsearch/%s.txt' % model_name, 'w')
    train_idxs = np.arange(n_train)
    min_rank = np.Inf
    patience = 3
    imp_threshold = 0.9
    patience_increase = 1.8
    fwrite("Starting %s...\n" % model_name)
    f.write("%s\n\n" % model_name)
    f.flush()
    sys.stdout.flush()
    for epoch in xrange(30):
        test_rank = model.test(Xim_test, Xtxt_test, epoch, saving_path=False)
        f.write('\tTest median rank = %d\n' % test_rank)
        f.flush()
        if test_rank < min_rank:
            if test_rank < imp_threshold * min_rank:
                patience = max(patience, epoch*patience_increase)
            min_rank = test_rank
        if patience<epoch:
            break
        
        f.write('Epoch: %d, W_im = %f, W_txt = %f\n' % (epoch,(model.W_img**2).mean(), 
                                                       (model.W_txt**2).mean()))
        f.flush()
        np.random.shuffle(train_idxs)
        Xim_train = Xim_train[train_idxs]
        Xtxt_train = Xtxt_train[train_idxs]
        tic = time()
        model.train(Xim_train, Xtxt_train, K,lr_img,lr_txt, batch_size=batch_size)
        toc = time()-tic
        f.write('\tTime = %fs\n' % toc)
        f.flush()
    res = {"lr_img":lr_img,"lr_txt":lr_txt,"K":K, "batch_size":batch_size, "epochs":epoch, "min_rank":min_rank}
    f.write(json.dumps(res))
    f.flush()
    f.close()
    fwrite("%s : %d epochs, %d median rank\n" % (model_name, epoch, min_rank))
    sys.stdout.flush()
    
if __name__=='__main__':
    np.random.seed(1234)
    which_set = 'r'
    data = np.load(which_set+'_similarity_data').all()
    
    d = data[data.keys()[0]]
    dim_img = d['image_emb'].shape[0]
    dim_txt = d['text_emb'].shape[1]
    dim_multi = 150
    
    image_embeddings = []
    text_embeddings = []
    product_ids = []
    product_ids = data.keys()
    for idx in product_ids:
        d = data.pop(idx)
        text_embeddings.append(d['text_emb'].mean(axis=0).astype(dtype))
        image_embeddings.append(d['image_emb'].astype(dtype))
    del data
    n_data = len(image_embeddings)
    n_data = 200
    product_ids = np.array(product_ids[:n_data])
    
    image_embeddings = np.array(image_embeddings[:n_data])
    text_embeddings = np.array(text_embeddings[:n_data])
    
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
    lr_imgs = [0.001, 0.01, 0.1, 1., 10.]
    lr_txts = [0.001, 0.01, 0.1, 1., 10.]
    Ks = [5, 10, 15]
    batch_sizes = [16, 32, 64]
    iter_params = product(lr_imgs, lr_txts, Ks, batch_sizes)
    Parallel(n_jobs=10)(delayed(run_gristep)(parameters, Xim_train, Xtxt_train) for parameters in iter_params)