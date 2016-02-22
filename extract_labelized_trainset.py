# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:39:52 2016

@author: hedi
Takes r_similarity_data or t_similarity_data, and extracts a subset out of it
in order to make a labelized training set with triplets (image, text, label)
"""

import numpy as np
import json
from collections import Counter
from sklearn.cross_validation import StratifiedShuffleSplit
if __name__ == '__main__':
    data = np.load('r_similarity_data').all()
    train_path = 'r_labelized_train.txt'
    test_path = 'r_labelized_test.txt'
    categorie1, categorie2, categorie3 = {}, {}, {}
    for idx,p in data.iteritems():
        cat1 = json.loads(p['product'])['Categorie1']
        cat2 = json.loads(p['product'])['Categorie2']
        cat3 = json.loads(p['product'])['Categorie3']
        categorie1[idx] = cat1
        categorie2[idx] = cat2
        categorie3[idx] = cat3
        
    ct = Counter(categorie3.values())
    labels = dict((k,v) for k,v in categorie3.iteritems() if ct[v]>15)
    product_ids = np.array(labels.keys())
    labels = np.array(labels.values())
    strat = StratifiedShuffleSplit(labels, train_size=2000)
    for train_index, test_index in strat:
        X = product_ids[train_index]
        Y = labels[train_index]
    
    strat = StratifiedShuffleSplit(Y)
    for train_index, test_index in strat:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
    with open(train_path, 'w') as f:
        f.write('Product_id Categorie1 Categorie2 Categorie3\n')
        for x,y in zip(X_train, y_train):
            f.write('%s %s %s %s\n' % (x, categorie1[x], categorie2[x], y))
            
    with open(test_path, 'w') as f:
        f.write('Product_id Categorie1 Categorie2 Categorie3\n')
        for x,y in zip(X_test, y_test):
            f.write('%s %s %s %s\n' % (x, categorie1[x], categorie2[x], y))