# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:53:22 2016

@author: hedi

From my format to Sylvain's
"""

import numpy as np
import sys
import cPickle
import os
fwrite = sys.stdout.write

if __name__ == '__main__':
    
    labels = []
    text_embeddings = []
    img_embeddings = []
        
        
    dataset = np.load('r_similarity_data').all()
    features_img = []
    features_txt = []
    image_paths = []
    product_ids = []
    N = len(dataset)
    fwrite('Creating feature matrices... ')
    for idx,d in dataset.iteritems():
        features_img.append(d['image_emb'])
#        image_paths.append(d['image_path'])
        features_txt.append(d['text_emb'])
        product_ids.append(idx)
    features_img = np.array(features_img)
    if features_img.shape[1] == 4096:
        features_img = features_img[:,:-1]
    fwrite('Done\n')
    
    del dataset
    
    model_path = os.path.join('plots/t_sim/','model.mod')
    W_img, W_txt,l2 = cPickle.load(open(model_path,'r'))
    r_features_img = np.dot(features_img, W_img)
    
    L = [x.shape[0] for x in features_txt]
    ends = np.cumsum(L)
    features_txt_c = np.concatenate(features_txt)
    r_features_txt_c = np.dot(features_txt_c, W_txt)
    b=0
    r_features_txt = []
    del features_txt_c
    for e in ends:
        r_features_txt.append(r_features_txt_c[b:e].mean(axis=0))
        b=e
    del r_features_txt_c
    r_features_txt = np.array(r_features_txt)
    
    features_product = 0.5*(r_features_img / 50. + r_features_txt)
    
#    labels_file = open('labels.txt', 'w')
#    prod_emb_file = open('m_prod_embeddings.txt', 'w')
#    for l, i in zip(product_ids, features_product):
#        labels_file.write(l + '\n')
#        prod_emb_file.write(' '.join(str(x) for x in i) + '\n')
#    labels_file.close()
#    prod_emb_file.close()
