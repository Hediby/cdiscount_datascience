# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 20:17:43 2016

@author: hedi
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import accuracy_score
from tokenizer import word_tokenize
from collections import Counter
import json
import sys
fwrite = sys.stdout.write

if __name__ == '__main__':
    # Labels
    trainset = open('r_labelized_train.txt','r')
    ids_train = []
    y_trains = [[], [], []]
    _ = trainset.readline()
    for line in trainset.readlines():
        L = line.strip().split(' ')
        ids_train.append(L[0])
        for cat in range(3):
            y_trains[cat].append(L[1+cat])
    trainset.close()
    
    testset = open('r_labelized_test.txt','r')
    ids_test = []
    y_tests = [[],[],[]]
    _ = testset.readline()
    for line in testset.readlines():
        L = line.strip().split(' ')
        ids_test.append(L[0])
        for cat in range(3):
            y_tests[cat].append(L[1+cat])
    testset.close()

    data = np.load('r_similarity_data').all()
    
    # Feature Extraction
    tf_train = []
    df = Counter()
    X_train_img = []
    for idx in ids_train:
        product = json.loads(data[idx]['product'])
        X_train_img.append(data[idx]['image_emb'])
        description = product['Description']
        tokenized = word_tokenize(description)
        tfs = {}
        for w,c in Counter(tokenized).iteritems(): 
            tfs[w] = float(c) / len(tokenized)
        for w in set(tokenized):
            df[w] += 1
        tf_train.append(tfs)
    X_train_img = np.array(X_train_img)
    D = len(ids_train)
    idfs = dict((k, np.log(float(D)/df[k])) for k in df)
    del df
    vocab = idfs.keys()
    vocab_dict = dict((k,v) for v,k in enumerate(vocab))
    vocab_size = len(vocab)
    
    X_train_txt = []
    for tf in tf_train:
        tfidf = np.zeros(vocab_size)
        for w in tf:
            tfidf[vocab_dict[w]] = tf[w] * idfs[w]
        X_train_txt.append(tfidf)
    X_train_txt = np.array(X_train_txt)
    
    X_test_txt = []
    X_test_img = []
    for idx in ids_test:
        product = json.loads(data[idx]['product'])
        X_test_img.append(data[idx]['image_emb'])
        description = product['Description']
        tokenized = word_tokenize(description)
        tfidf = np.zeros(vocab_size)
        for w,c in Counter(tokenized).iteritems():
            if w in vocab:
                tfidf[vocab_dict[w]] = idfs[w] * float(c) / len(tokenized)
        X_test_txt.append(tfidf)
    X_test_txt = np.array(X_test_txt)
    X_test_img = np.array(X_test_img)
    
    # Training
    for cat, (y_train, y_test) in enumerate(zip(y_trains, y_tests)):
        lr_txt = LogisticRegression()
        lr_img = LogisticRegression()
        lr_txt.fit(X_train_txt, y_train)
        lr_img.fit(X_train_img, y_train)
        classes = lr_img.classes_
        p_txt = lr_txt.predict_proba(X_train_txt)
        p_img = lr_img.predict_proba(X_train_img)
        p = p_img + p_txt
        train_score = 100*accuracy_score(classes[p.argmax(axis=1)], y_train)    
        p_txt = lr_txt.predict_proba(X_test_txt)
        p_img = lr_img.predict_proba(X_test_img)
        p = p_img + p_txt
        test_score = 100*accuracy_score(classes[p.argmax(axis=1)], y_test)
        fwrite('Category %d:\n\tTrain score = %2.1f%%\n\tTest score = %2.1f%%\n\n' % (cat+1,train_score, test_score))
