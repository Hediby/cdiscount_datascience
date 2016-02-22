# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:56:11 2016

@author: hedi


"""
import numpy as np
import sys
from time import time
import os
from sim_model import m_sim_mean_map, m_sim_mean
import theano

fwrite = sys.stdout.write
dtype = theano.config.floatX

if __name__=='__main__':
    np.random.seed(1234)
    which_set = 'r'
    fwrite('Loading data...')
    data = np.load(which_set+'_similarity_data').all()
    fwrite(' Ok\n')
    pooling = 'mean' #mean, max or softmax
    folder = os.path.join(which_set+'_sim', pooling)
    if not os.path.exists(folder):
        os.mkdir(folder)
    saving_path = os.path.join(folder, "model.mod")
    
    d = data[data.keys()[0]]
    dim_img = d['image_emb'].shape[0]
    dim_txt = d['text_emb'].shape[1]
    dim_multi = 150
    
    image_embeddings = []
    text_embeddings = []
    product_ids = []
    fwrite('Loading data ...\n')
    product_ids = data.keys()
    for idx in product_ids:
        d = data.pop(idx)
        if not len(data)%5000:
            fwrite('%d\n' % len(data))
        text_embeddings.append(d['text_emb'].mean(axis=0).astype(dtype))
        image_embeddings.append(d['image_emb'].astype(dtype))
    del data
    n_data = len(image_embeddings)
    fwrite('Done\n')
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
    lr_img = 0.001
    lr_txt = 0.0001
    K = 10
    batch_size = 32
    
    fwrite('dim_img = %d, dim_txt = %d, dim_multi = %d\n' % (dim_img, dim_txt, dim_multi))
    model = m_sim_mean.create(dim_img, dim_txt, dim_multi)
    train_idxs = np.arange(n_train)

    for epoch in range(51):
        if not epoch % 5:
            test_rank = model.test(Xim_test, Xtxt_test, epoch=0, saving_path = folder)
            train_rank = model.test(Xim_train, Xtxt_train, epoch=0, saving_path = False)
            fwrite('\tTest median rank = %d\n' % test_rank)
            fwrite('\tTrain median rank = %d\n' % train_rank)
            sys.stdout.flush()
        
        
        fwrite('Epoch: %d, W_im = %f, W_txt = %f\n' % (epoch,(model.W_img**2).mean(), 
                                                       (model.W_txt**2).mean()))

        np.random.shuffle(train_idxs)
        Xim_train = Xim_train[train_idxs]
        Xtxt_train = Xtxt_train[train_idxs]
        
        tic = time()
        model.train(Xim_train, Xtxt_train,  K,lr_img,lr_txt, 
                    batch_size=batch_size, verbose=True)
        toc = time()-tic
        model.save(saving_path)
        fwrite('\tTime = %fs\n' % toc)
