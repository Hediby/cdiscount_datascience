# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:01:56 2016

@author: hedi

We load the images embedding
 - r_img_caffe_features.dat : reduced version
 - t_img_caffe_features.dat : total version
Then we select some of them, and look for most similar ones in the origin space.

Next, we project all these embeddings in the new multimodal space, and look 
for most similar in that space.
"""

import numpy as np
from logistic_classifier import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cPickle
#import seaborn as sns
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
fwrite = sys.stdout.write

def get_closest(X, Y, batch_size=500, k=5):
    """
    Returns, for each line of X, the indexes and similarities of the k most similar
    lines of Y.
    """
    N = X.shape[0]
    n_batches = N/batch_size + 1*(not 0==N%batch_size)
    args = []
    sims = []
    fwrite('\n')
    for i in np.arange(n_batches):
        if not i%(n_batches/10):
            fwrite("\tBatch %d/%d\n" % (i,n_batches))
        begin = i*batch_size
        end = min(N,(i+1)*batch_size)
        x = X[begin:end]
        M = cosine_similarity(x,Y)
        for m in M:
            idxs = np.argsort(-m)[1:k+1]
            args.append(idxs)
            sims.append(m[idxs])
    return args, sims

if __name__ == '__main__':
    
    
    folder = 'plots/r_sim/simi'
    model_path = os.path.join('plots/t_sim/','model.mod')
    dataset = np.load('r_similarity_data').all()
    
    images_path = 'images/img/training'
    break_all = False
    K = 0
    fwrite('Retrieving image paths... ')
    for (dirpath, dirnames, filenames) in os.walk(images_path):
        for f in filenames:
            idx = f.split('.')[0]
            if idx in dataset:
                dataset[idx]['image_path'] = os.path.join(dirpath,f)
                K += 1
            if K == len(dataset):
                break_all = True
            if break_all:
                break
        if break_all:
            break
    fwrite('Done\n')
    features_img = []
    features_txt = []
    image_paths = []
    product_ids = []
    N = len(dataset)
    fwrite('Creating feature matrices... ')
    for idx,d in dataset.iteritems():
        features_img.append(d['image_emb'])
        image_paths.append(d['image_path'])
        features_txt.append(d['text_emb'])
        product_ids.append(idx)
    features_img = np.array(features_img)
    if features_img.shape[1] == 4096:
        features_img = features_img[:,:-1]
    fwrite('Done\n')
    
    del dataset
    
    W_img, W_txt,l2 = cPickle.load(open(model_path,'r'))
    r_features_img = np.dot(features_img, W_img)
    fwrite('Text features... ')
    L = [x.shape[0] for x in features_txt]
    ends = np.cumsum(L)
    fwrite('Concatenate...')
    features_txt_c = np.concatenate(features_txt)
    fwrite('Done\n Pooling features...')
    b=0
    features_txt = []
    
    for e in ends:
        features_txt.append(features_txt_c[b:e].mean(axis=0))
        b=e
    del features_txt_c
    features_txt = np.array(features_txt)
    r_features_txt = np.dot(features_txt, W_txt)
    fwrite('Done\n')
    k = 5
    N_to_show = 20
    np.random.seed(1578)
    indexes = np.random.choice(N, size=N_to_show, replace=False)
    
    test_images = features_img[indexes]
    r_test_images = r_features_img[indexes]
    r_test_texts = r_features_txt[indexes]
    
#==============================================================================
# Intra-modality similarities with unimodal models
#==============================================================================

    # Image
    fwrite('Image in CNN space... ')
    S = cosine_similarity(features_img[indexes], features_img)
    A = np.argsort(-S)[:,:k]
    
    fig = plt.figure(figsize=(k*6,N_to_show*4))
    fig.patch.set_facecolor('black')
    gs1 = gridspec.GridSpec(N_to_show, k+1)
    gs1.update(wspace=1., hspace=1.)
    for i in np.arange(N_to_show):
        I = plt.imread(image_paths[indexes[i]])
        plt.subplot(N_to_show, k+1, 1+i*(k+1), frameon=False)
        plt.imshow(I)
        for j in np.arange(k):
            a = A[i][j]
            I = plt.imread(image_paths[a])
            plt.subplot(N_to_show, k+1, 2+i*(k+1)+j, frameon=False)
            plt.imshow(I)
    for ax in fig.axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(folder,'simi_origin_img.png'), facecolor=fig.get_facecolor())
    fwrite('Done\n')
    
    # Text
    fwrite('Text in embeddings space... ')
    S = cosine_similarity(features_txt[indexes], features_txt)
    A = np.argsort(-S)[:,:k]
    
    fig = plt.figure(figsize=(k*6,N_to_show*4))
    fig.patch.set_facecolor('black')
    gs1 = gridspec.GridSpec(N_to_show, k+1)
    gs1.update(wspace=1., hspace=1.)
    for i in np.arange(N_to_show):
        I = plt.imread(image_paths[indexes[i]])
        plt.subplot(N_to_show, k+1, 1+i*(k+1), frameon=False)
        plt.imshow(I)
        for j in np.arange(k):
            a = A[i][j]
            I = plt.imread(image_paths[a])
            plt.subplot(N_to_show, k+1, 2+i*(k+1)+j, frameon=False)
            plt.imshow(I)
    for ax in fig.axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(folder,'simi_origin_txt.png'), facecolor=fig.get_facecolor())
    fwrite('Done\n')
#    
##==============================================================================
## Intra-modality similarities with multimodal models
##==============================================================================
    
    #Image
    fwrite('Image in multimodal space... ')
    S = cosine_similarity(r_features_img[indexes], r_features_img)
    A = np.argsort(-S)[:,:k]
    
    fig = plt.figure(figsize=(k*6,N_to_show*4))
    fig.patch.set_facecolor('black')
    gs1 = gridspec.GridSpec(N_to_show, k+1)
    gs1.update(wspace=1., hspace=1.)
    for i in np.arange(N_to_show):
        I = plt.imread(image_paths[indexes[i]])
        plt.subplot(N_to_show, k+1, 1+i*(k+1), frameon=False)
        plt.imshow(I)
        for j in np.arange(k):
            a = A[i][j]
            I = plt.imread(image_paths[a])
            plt.subplot(N_to_show, k+1, 2+i*(k+1)+j, frameon=False)
            plt.imshow(I)
    for ax in fig.axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(folder,'simi_m_img.png'), 
                facecolor=fig.get_facecolor())
    fwrite('Done\n')
    
    #Text
    fwrite('Text in multimodal space... ')
    S = cosine_similarity(r_features_txt[indexes], r_features_txt)
    A = np.argsort(-S)[:,:k]
    
    fig = plt.figure(figsize=(k*6,N_to_show*4))
    fig.patch.set_facecolor('black')
    gs1 = gridspec.GridSpec(N_to_show, k+1)
    gs1.update(wspace=1., hspace=1.)
    for i in np.arange(N_to_show):
        I = plt.imread(image_paths[indexes[i]])
        plt.subplot(N_to_show, k+1, 1+i*(k+1), frameon=False)
        plt.imshow(I)
        for j in np.arange(k):
            a = A[i][j]
            I = plt.imread(image_paths[a])
            plt.subplot(N_to_show, k+1, 2+i*(k+1)+j, frameon=False)
            plt.imshow(I)
    for ax in fig.axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(folder,'simi_m_text.png'), 
                facecolor=fig.get_facecolor())
    fwrite('Done\n')
#==============================================================================
# Product retrieval
#==============================================================================
    features_product = 0.5*(r_features_img/50. + r_features_txt)
    fwrite('Products... ')
    S = cosine_similarity(features_product[indexes], features_product)
    A = np.argsort(-S)[:,:k]
    fig = plt.figure(figsize=(k*6,N_to_show*4))
    fig.patch.set_facecolor('black')
    gs1 = gridspec.GridSpec(N_to_show, k+1)
    gs1.update(wspace=1., hspace=1.)
    for i in np.arange(N_to_show):
        I = plt.imread(image_paths[indexes[i]])
        plt.subplot(N_to_show, k+1, 1+i*(k+1), frameon=False)
        plt.imshow(I)
        for j in np.arange(k):
            a = A[i][j]
            I = plt.imread(image_paths[a])
            plt.subplot(N_to_show, k+1, 2+i*(k+1)+j, frameon=False)
            plt.imshow(I)
    for ax in fig.axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(folder,'simi_product.png'), 
                facecolor=fig.get_facecolor())
    fwrite('Done\n')
