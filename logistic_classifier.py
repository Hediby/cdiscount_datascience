# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:17:29 2015

@author: hedibenyounes
"""

import theano
import theano.tensor as T
import numpy as np
import cPickle
dtype = theano.config.floatX
dtype = 'float32'
class Logistic(object):
    def __init__(self, n_in, n_classes, l2 = None):
        # Model
        W = 0.01*np.random.randn(n_in, n_classes).astype(dtype)
        b = 0.01*np.random.randn(n_classes).astype(dtype)
        self.W = theano.shared(W, name='W')
        self.b = theano.shared(b, name='b')
        self.params = [self.W, self.b]
        
        self.input = T.fmatrix('input')
        self.y_true = T.ivector('y_true')
        self.y_hat = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        
        # Train
        self.loglikelihood = -T.log(self.y_hat[T.arange(self.y_hat.shape[0]),self.y_true])
        self.cost = T.mean(self.loglikelihood)
        if l2:
            for p in self.params:
                self.cost += l2*T.sum(p**2)
        self.gradients = T.grad(self.cost, self.params)
        self.lr = T.fscalar('lr')
        updates = [(p,p-self.lr*g) for p,g in zip(self.params, self.gradients)]
        
        self.train = theano.function(inputs=[self.input, self.y_true, self.lr], 
                                     outputs=self.cost, 
                                     updates = updates,
                                     allow_input_downcast=True)
                                     
        # Predict
        self.y_predict = T.argmax(self.y_hat, axis=1)
        self.predict = theano.function(inputs=[self.input],
                                       outputs=self.y_predict,
                                       allow_input_downcast=True)
                                       
class Model(object):
    def __init__(self):
        self.params = []
        self.l2 = None
    def save(self, path):
        to_save = {'params':[p.get_value() for p in self.params],
                             'l2':self.l2}
        cPickle.dump(to_save,open(path,'w'))
        return 'Done'
                                       
class m_sum_projection(Model):
    def __init__(self, n_text, n_image, n_classes, l2 = None):
        # Model
        W_image = 0.1*np.random.randn(n_image, n_text).astype(dtype)
        W = 0.1*np.random.randn(n_text, n_classes).astype(dtype)
        b = 0.1*np.random.randn(n_classes).astype(dtype)
        self.W_image = theano.shared(W_image, name='W_image')
        self.W = theano.shared(W, name='W')
        self.b = theano.shared(b, name='b')
        self.params = [self.W_image, self.W, self.b]
        
        self.text = T.fmatrix('text')
        self.image = T.fmatrix('image')
        self.embedded_image = T.dot(self.image, self.W_image)
        self.y_true = T.ivector('y_true')
        self.y_hat = T.nnet.softmax(T.dot(self.text + self.embedded_image, self.W) + self.b)
        
        # Train
        self.loglikelihood = -T.log(self.y_hat[T.arange(self.y_hat.shape[0]),self.y_true])
        self.cost = T.mean(self.loglikelihood)
        if l2:
            for p in self.params:
                self.cost += l2*T.sum(p**2)
        self.gradients = T.grad(self.cost, self.params)
        self.lr = T.fscalar('lr')
        updates = [(p,p-self.lr*g) for p,g in zip(self.params, self.gradients)]
        
        self.train = theano.function(inputs=[self.image, self.text, 
                                             self.y_true, self.lr], 
                                     outputs=self.cost, 
                                     updates = updates,
                                     allow_input_downcast=True)
                                     
        # Predict
        self.y_predict = T.argmax(self.y_hat, axis=1)
        self.predict = theano.function(inputs=[self.image, self.text],
                                       outputs=self.y_predict,
                                       allow_input_downcast=True)
            

class m_both_projection(Model):
    def __init__(self, n_text, n_image, n_hid, n_classes, l2 = None):
        # Model
        W_image = 0.01*np.random.randn(n_image, n_hid).astype(dtype)
        W_text = 0.01*np.random.randn(n_text, n_hid).astype(dtype)
        W = 0.01*np.random.randn(n_hid, n_classes).astype(dtype)
        b = 0.1*np.random.randn(n_classes).astype(dtype)
        self.l2 = l2
        self.W_image = theano.shared(W_image, name='W_image')
        self.W_text = theano.shared(W_text, name='W_text')
        self.W = theano.shared(W, name='W')
        self.b = theano.shared(b, name='b')
        self.params = [self.W_image, self.W_text, self.W, self.b]
        
        self.text = T.fmatrix('text')
        self.image = T.fmatrix('image')
        self.embedded_image = T.dot(self.image, self.W_image)
        self.embedded_text = T.dot(self.text, self.W_text)
        self.multimodal_emb = T.tanh(self.embedded_image + self.embedded_text)
        self.y_true = T.ivector('y_true')
        self.y_hat = T.nnet.softmax(T.dot(self.multimodal_emb, self.W) + self.b)
        
        # Train
        self.loglikelihood = -T.log(self.y_hat[T.arange(self.y_hat.shape[0]),self.y_true])
        self.cost = T.mean(self.loglikelihood)
        if self.l2:
            for p in self.params:
                self.cost += self.l2*T.sum(p**2)
        self.gradients = T.grad(self.cost, self.params)
        self.lr = T.fscalar('lr')
        updates = [(p,p-self.lr*g) for p,g in zip(self.params, self.gradients)]
        
        self.train = theano.function(inputs=[self.image, self.text, 
                                             self.y_true, self.lr], 
                                     outputs=self.cost, 
                                     updates = updates,
                                     allow_input_downcast=True)
                                     
        # Predict
        self.y_predict = T.argmax(self.y_hat, axis=1)
        self.predict = theano.function(inputs=[self.image, self.text],
                                       outputs=self.y_predict,
                                       allow_input_downcast=True)
                                       
if __name__=='__main__':       
    n_batch = 15
    n_in = 10
    n_classes = 5
    model = Logistic(n_in,n_classes)
    x_in = np.random.rand(n_batch,n_in)
    y_true = np.random.randint(0,n_classes, size=n_batch)
    lr = 0.05
    for i in range(1000):
        cost = model.train(x_in, y_true, lr)
        if not i % 50:
            print cost
            print "predict :\t", model.predict(x_in), "\ntrue :\t", y_true
            print '\n'