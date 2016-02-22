# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:13:56 2016

@author: hedi

This files learns to classify products according to 3 criterions :
- only textual features
- only visual features
- concatenation of textual and visual features
"""

import numpy as np
from logistic_classifier import Logistic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
fwrite = sys.stdout.write

def train_model(model, X_train, y_train, X_test, y_test,
                lr=0.1, n_epoch=500, batch_size=64, 
                title='', filename='picture'):
    train_precisions = []
    test_precisions = []
    fwrite('Train...\n')
    for epoch in range(n_epoch):
        train_indexes = np.arange(n_train)
        np.random.shuffle(train_indexes)
        X_train = X_train[train_indexes]
        y_train = y_train[train_indexes]
        if not epoch % 5:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            train_precision = np.mean(y_train == y_pred_train)
            train_precisions += [train_precision]
            test_precision = np.mean(y_test == y_pred_test)
            test_precisions += [test_precision]
            w,b = [p.get_value().mean() for p in model.params]
            fwrite('Epoch %d/%d : train = %1.3f, test = %1.3f\n' % (epoch,
                                                                    n_epoch,
                                                                    train_precision, 
                                                                    test_precision))

            fwrite('\t')
            for p in model.params:
                fwrite('%s = %f ' % (p, (p.get_value()**2).sum()))
            fwrite('\n')
            sys.stdout.flush()
        for i in range(n_train/batch_size):
            begin = i*batch_size
            end = min((i+1)*batch_size, n_train)
            x = X_train[begin:end]
            y = y_train[begin:end]
            cost = model.train(x,y,lr)
    fwrite('\n')

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
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
    features = []
    labels = []
    fwrite('Extract features ...')
    for d in data.itervalues():
        feat = np.concatenate((d['text_emb'], d['image_emb']))
        label = d[category]
        labels.append(label)
        features.append(feat)
#        if len(labels)>5000:
#            break
    del data
    features = np.array(features)
    features = features.astype('float32')
    fwrite('Done\n')
    clean2dirty = list(set(labels))
    dirty2clean = dict([(k,v) for v,k in enumerate(clean2dirty)])
    labels = [dirty2clean[v] for v in labels]
    labels = np.array(labels)
    np.random.seed(1234)
    
    # Removing constant columns
    fwrite('Before removal : %s\n' % str(features.shape))
    sys.stdout.flush()
    Z = features.var(axis=0)
    remove_idxs = []
    for idx, z in enumerate(Z):
        if z == 0:
            remove_idxs += [idx]
    features = np.delete(features, remove_idxs, 1)
    fwrite('After removal : %s\n' % str(features.shape))
    sys.stdout.flush()
    
        
    n = features.shape[0]
    n_classes = len(clean2dirty)
    
    indexes = np.arange(n)
    np.random.shuffle(indexes)
    
    features = features[indexes]
    labels = labels[indexes]
    n_train = int(0.66*n)
    
    for filename in [ 'only_image', 'only_text','multimodal_concat']:
        if filename == 'only_text':
#            n_features = (features[:,:200] - features[:,:200].mean(axis=0)) / np.sqrt(features[:,:200].var(axis=0))
            n_features = features[:,:200]
            title = 'Only text'
            if category=='Categorie1':
                lr = 1.
                l2 = 0.000001
            elif category=='Categorie2':
                lr = 1.
                l2 = 0.000001
            elif category=='Categorie3':
                lr = 1.
                l2 = 0.000001
        elif filename == 'only_image':
#            n_features = (features[:,201:] - features[:,201:].mean(axis=0)) / np.sqrt(features[:,201:].var(axis=0))
            n_features = features[:,201:]            
            title = 'Only image'
            if category=='Categorie1':
                lr = 0.001
                l2 = .1
            elif category=='Categorie2':
                lr = 0.001
                l2 = .01
            elif category=='Categorie3':
                lr = 0.001
                l2 = .01
        elif filename == 'multimodal_concat':
#            n_features = (features - features.mean(axis=0)) / np.sqrt(features.var(axis=0))
            n_features = features
            title = 'Concatenation of text and image'
            if category=='Categorie1':
                lr = 0.01
                l2 = .01
            elif category=='Categorie2':
                lr = 0.01
                l2 = .01
            elif category=='Categorie3':
                lr = 0.01
                l2 = .1
        else:
            raise Exception('Unimplemented filename %s' % filename)
        fwrite(title + '\n')
            
        X_train, y_train = n_features[:n_train], labels[:n_train]
        X_test, y_test = n_features[n_train+1:], labels[n_train+1:]
        
        n_epoch = 200
        batch_size = 64
        n_in = n_features.shape[1]
        model = Logistic(n_in, n_classes, l2=l2)
    
        train_model(model, X_train, y_train, X_test, y_test, 
                    lr=lr, n_epoch=n_epoch, batch_size=batch_size,
                    title=title, filename=filename)
