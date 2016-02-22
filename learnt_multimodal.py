# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:29:06 2016

@author: hedi
"""

import numpy as np
from logistic_classifier import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
fwrite = sys.stdout.write

def train_model(model, Xim_train, Xtxt_train, y_train, Xim_test, Xtxt_test, y_test,
                lr=0.1, n_epoch=500, batch_size=64, 
                title='', filename='picture', saving_path='saved'):
    train_precisions = []
    test_precisions = []
    fwrite('Train...\n')
    for epoch in range(n_epoch):
        train_indexes = np.arange(n_train)
        np.random.shuffle(train_indexes)
        Xim_train = Xim_train[train_indexes]
        Xtxt_train = Xtxt_train[train_indexes]
        y_train = y_train[train_indexes]
        if not epoch % 5:
            y_pred_train = model.predict(Xim_train, Xtxt_train)
            y_pred_test = model.predict(Xim_test, Xtxt_test)
            train_precision = np.mean(y_train == y_pred_train)
            train_precisions += [train_precision]
            test_precision = np.mean(y_test == y_pred_test)
            test_precisions += [test_precision]
            fwrite('Epoch %d/%d : train = %1.3f, test = %1.3f\n' % (epoch,
                                                                    n_epoch,
                                                                    train_precision, 
                                                                    test_precision))

            fwrite('\t')
            for p in model.params:
                fwrite('%s = %f ' % (p, (p.get_value()**2).sum()))
            fwrite('\n')
            sys.stdout.flush()
            cPickle.dump(model, open(saving_path,'w'))
            
        for i in range(n_train/batch_size):
            begin = i*batch_size
            end = min((i+1)*batch_size, n_train)
            xim = Xim_train[begin:end]
            xtxt = Xtxt_train[begin:end]
            y = y_train[begin:end]
            cost = model.train(xim,xtxt,y,lr)
    fwrite('\n')
    cPickle.dump(model, open(saving_path,'w'))
    y_pred_train = model.predict(Xim_train, Xtxt_train)
    y_pred_test = model.predict(Xim_test, Xtxt_test)
    train_precision = np.mean(y_train == y_pred_train)
    test_precision = np.mean(y_test == y_pred_test)
    
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlim(0,n_epoch+1)
    ax.set_ylim(0,1.1)

    ax.plot(np.linspace(0,epoch, len(train_precisions)), 
               train_precisions, '.-',
               c='b', linewidth=0.7,
               label='Train precision (%1.3f)' % train_precision)
    ax.plot(np.linspace(0,epoch, len(test_precisions)), 
               test_precisions, '.-',
               c='r', linewidth=0.7,
               label='Test precision (%1.3f)' % test_precision)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.savefig('plots/%s/%s' % (category,filename))
    
if __name__ == "__main__":
    data = np.load('multimodal_dataset.npy').all()
    category = 'Categorie3'
    features_txt = []
    features_im = []
    labels = []
    fwrite('Extract features ...')
    for d in data.itervalues():
        features_txt.append(d['text_emb'])
        features_im.append(d['image_emb'])
        label = d[category]
        labels.append(label)
#        if len(labels)>5000:
#            break
    del data
    features_im = np.array(features_im)
    features_txt = np.array(features_txt)
    features_im = features_im.astype('float32')
    features_txt = features_txt.astype('float32')
    fwrite('Done\n')
    clean2dirty = list(set(labels))
    dirty2clean = dict([(k,v) for v,k in enumerate(clean2dirty)])
    labels = [dirty2clean[v] for v in labels]
    labels = np.array(labels)
    np.random.seed(1234)
    
    # Removing constant columns
    fwrite('Before removal (image): %s\n' % str(features_im.shape))
    sys.stdout.flush()
    Z = features_im.var(axis=0)
    remove_idxs = []
    for idx, z in enumerate(Z):
        if z == 0:
            remove_idxs += [idx]
    features_im = np.delete(features_im, remove_idxs, 1)
    fwrite('After removal (image): %s\n' % str(features_im.shape))
    sys.stdout.flush()
    
    fwrite('Before removal (text): %s\n' % str(features_txt.shape))
    sys.stdout.flush()
    Z = features_txt.var(axis=0)
    remove_idxs = []
    for idx, z in enumerate(Z):
        if z == 0:
            remove_idxs += [idx]
    features_txt = np.delete(features_txt, remove_idxs, 1)
    fwrite('After removal (text): %s\n' % str(features_txt.shape))
    sys.stdout.flush()
        
    n = len(labels)
    n_classes = len(clean2dirty)
    
    indexes = np.arange(n)
    np.random.shuffle(indexes)
    
    features_im = features_im[indexes]
    features_txt = features_txt[indexes]
    labels = labels[indexes]
    n_train = int(0.66*n)
    n_features_im = (features_im - features_im.mean(axis=0)) / np.sqrt(features_im.var(axis=0))
    n_features_txt = (features_txt - features_txt.mean(axis=0)) / np.sqrt(features_txt.var(axis=0))
        
    Xim_train, Xtxt_train, y_train = n_features_im[:n_train], n_features_txt[:n_train], labels[:n_train]
    Xim_test, Xtxt_test, y_test = n_features_im[n_train+1:], n_features_txt[n_train+1:], labels[n_train+1:]
    if category == 'Categorie1':
        lr = .01
        l2 = 0.01    
    elif category == 'Categorie2':
        lr = .01
        l2 = 0.1
    elif category == 'Categorie3':
        lr = .1
        l2 = 0.01
    n_epoch = 200
    batch_size = 64
    n_text = n_features_txt.shape[1]
    n_image = n_features_im.shape[1]
#    model = m_sum_projection(n_text, n_image, n_classes, l2=.1)
    model = m_both_projection(n_text, n_image=n_image, n_hid=300, n_classes=n_classes, l2=l2)
    saving_path = 'm_both_projection_%s.model' % category
    title = "Learnt both projections"
    filename = "learnt_both_proj"
    train_model(model, Xim_train, Xtxt_train, y_train, Xim_test, Xtxt_test,y_test, 
                lr=lr, n_epoch=n_epoch, batch_size=batch_size,
                title=title, filename=filename, saving_path=saving_path)
    
