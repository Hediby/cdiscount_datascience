# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:04:42 2016

@author: hedi
"""

import numpy as np
from sim_model import m_sim_mean
from goldberg.scripts.infer import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from os.path import join
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
fwrite = sys.stdout.write
      
      
if __name__ == '__main__':
    folder = 't_sim/mean'
    model_path = join(folder, 'model.mod')
    data_path = 'r_similarity_data'
    model = Embeddings('../product2vec2/embeddings/all/vecs.npy')
    multimodal = m_sim_mean.load(model_path)
    vocab = model._vocab
    features_words = model._vecs
    del model
    
    np.random.seed(1234)
    data = np.load(data_path).all()
    
    features_img = []
    image_paths = []
    products_id = []
    features_txt = []
    texts = []
    fwrite('Creating feature matrices... ')
    for idx, d in data.iteritems():
        features_img.append(d['image_emb'])
        image_paths.append(d['image_path'])
        features_txt.append(d['text_emb'].mean(axis=0))
        products_id.append(idx)
        texts.append(d['product'])
    del data
    indexes = np.arange(len(texts))
    np.random.shuffle(indexes)
    n_train = int(0.7*len(texts))
    indexes = indexes[n_train+1:]
    print len(indexes)
    n_imgs = 800
    indexes = indexes[:800]
    features_img = np.array(features_img)
    features_img = features_img[indexes]
    features_img = features_img / np.linalg.norm(features_img, 2, axis=1).reshape(-1,1)
    
    features_txt = np.array(features_txt)
    features_txt = features_txt[indexes]
    features_txt = features_txt / np.linalg.norm(features_txt, 2, axis=1).reshape(-1,1)
    
    _texts, _image_paths, _products_id = [],[],[]
    for i, (t, ip, p) in enumerate(zip(texts, image_paths, products_id)):
        if i in indexes:
            _texts.append(t)
            _image_paths.append(ip)
            _products_id.append(p)
    texts = _texts
    image_paths = _image_paths
    products_id = _products_id
    fwrite('Done\n')
    features_img = np.dot(features_img, multimodal.W_img)
    features_words = np.dot(features_words, multimodal.W_txt)
    features_txt = np.dot(features_txt, multimodal.W_txt)
    
    fwrite('features_txt : %s \nfeatures_img : %s\n' % (str(features_words.shape),
                                                      str(features_img.shape)))
    
    #%% Image tagging
#    plt.close('all')
#    sns.set_style("whitegrid", {'axes.grid' : False})
#    k = 5
#    index = np.random.choice(features_img.shape[0])
#    image = features_img[index]
#    
#    sims = cosine_similarity(image.reshape(1,-1), features_words)[0]
#    args = np.argsort(-sims)[:k]
#    sims = sims[args]
#    words = []
#    ld = json.loads(texts[index])
#    fwrite('Marque : %s\n' % ld['Marque'])
#    fwrite("Description:\n%s\n\nMots les plus proches de l'image:\n" % ld['Description'])
#    for idx, sim in zip(args, sims):
#        word = vocab[idx]
#        words.append(word)
#        fwrite('%s : %f\n' % (word, sim))
#    indexes = [vocab.index(w) for w in words]
#    word_features = features_words[indexes]
#    sims = cosine_similarity(word_features, features_img)
#    args = [np.argsort(-s)[:k] for s in sims]
#    sims = [s[a] for s,a in zip(sims, args)]
#    
#    N_to_show = len(words)
#    fig = plt.figure(figsize=(k*6,N_to_show*3))
#    gs1 = gridspec.GridSpec(N_to_show, k+1)
#    gs1.update(wspace=.5, hspace=.5)
#    for j,(w, arg, sim) in enumerate(zip(words, args, sims)):
#        for i,(a, s) in enumerate(zip(arg, sim)):
#            I = plt.imread(image_paths[a])
#            plt.subplot(N_to_show, k, j*k+i+1, frameon=False)
#            plt.imshow(I)
#            plt.title('%s : %f' % (w.decode('utf-8'),s), fontsize=u'small')
#    for ax in fig.axes:
#        ax.axes.get_xaxis().set_visible(False)
#        ax.axes.get_yaxis().set_visible(False)
#    fig.tight_layout()
#    
#    
#    fig = plt.figure()
#    I = plt.imread(image_paths[index])
#    plt.imshow(I)
#    plt.title('Product name = %s (id %d)' % (products_id[index], index))
#    
    #%% TSNE
    plt.close('all') 
    
    k = 2
    sims = cosine_similarity(features_img, features_words)
    args_im= np.argsort(-sims, axis=1)[:,:k]
    del sims
    args_im = set(args_im.flatten())
    sims = cosine_similarity(features_txt, features_words)
    args_txt = np.argsort(-sims, axis=1)[:,:k]
    del sims
    args_txt = set(args_txt.flatten())
    args = args_im.union(args_txt)
    args = np.array(list(args))
    fwrite("%d words for %d images\n" % (args.shape[0],n_imgs))
    
    to_reduce = np.concatenate((features_img, features_txt, features_words[args]))
    tsne = TSNE(verbose=5)
    pca = PCA(n_components=2)
    pca_reduced = pca.fit_transform(to_reduce)
    reduced = tsne.fit_transform(pca_reduced)
       
    fig = plt.figure()
#    plt.scatter(reduced[2*n_imgs+1:,0], reduced[1+2*n_imgs:,1], c='b', s=10,
#                label="Words", linewidth=0.1, alpha=0.6)
    plt.scatter(reduced[n_imgs+1:2*n_imgs,0], reduced[1+n_imgs:2*n_imgs,1], c='green', s=10,
                label="Texts", linewidth=0.1, alpha=0.6)
    plt.scatter(reduced[:n_imgs,0], reduced[:n_imgs,1], c='r', s=10,
                label="Images", linewidth=0.1, alpha=0.6)
    plt.legend()
    plt.savefig(join(folder,'tsne.png'))
    
    