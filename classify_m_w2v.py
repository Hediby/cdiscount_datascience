# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:56:57 2016

@author: hedi
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sim_model import m_sim_mean
import sys
fwrite = sys.stdout.write

if __name__ == '__main__':
    # Labels
    trainset = open('r_labelized_train.txt','r')
    multimodal_model_path = 't_sim_map/model.mod'
    model = m_sim_mean.load(multimodal_model_path)
    W_txt = model.W_txt
    del model
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
    X_train = []
    for idx in ids_train:
        X_train.append(data[idx]['text_emb'].mean(axis=0))
    X_train = np.array(X_train)
    X_train = np.dot(X_train, W_txt)
    X_test = []
    for idx in ids_test:
        X_test.append(data[idx]['text_emb'].mean(axis=0))
    X_test = np.array(X_test)
    X_test = np.dot(X_test, W_txt)
    
    # Training
    for cat, (y_train, y_test) in enumerate(zip(y_trains, y_tests)):
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        train_score = 100*lr.score(X_train, y_train)
        test_score = 100*lr.score(X_test, y_test)
        fwrite('Category %d:\n\tTrain score = %2.1f%%\n\tTest score = %2.1f%%\n\n' % (cat+1,train_score, test_score))
