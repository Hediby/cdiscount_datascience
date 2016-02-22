# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:00:43 2016

@author: hedi
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import theano
import theano.tensor as T
import cPickle
from os.path import join
from sklearn.metrics.pairwise import cosine_similarity
fwrite = sys.stdout.write
dtype = theano.config.floatX

epsilon = 1e-6

def cosine(x,y):
    return T.dot(x,y) / (x.norm(2) * y.norm(2, axis=0))
    
class Model(object):
    def __init__(self, W_img_val, W_txt_val, l2):
        self.l2 = l2
        self.W_img = W_img_val
        self.W_txt = W_txt_val
        self.n_img = self.W_img.shape[0]
        self.n_hid = self.W_img.shape[1]
        self.n_txt = self.W_txt.shape[0]
        self._initialize_model()

    @classmethod
    def create(cls, n_img, n_txt, n_hid, l2=None):
        W_img_val = 0.01 * np.random.randn(n_img, n_hid).astype(dtype)
        W_txt_val = 0.01 * np.random.randn(n_txt, n_hid).astype(dtype)
        return cls(W_img_val, W_txt_val, l2)
    
    @classmethod
    def load(cls,path):
        params = cPickle.load(open(path,'r'))
        W_img_val, W_txt_val = params
        l2 = None
        return cls(W_img_val, W_txt_val, l2=l2)
    
    def save(self, path):
        to_save = [self.W_img, self.W_txt]
        cPickle.dump(to_save, open(path,'w'))
        return None
        
    def _initialize_model(self):
        raise NotImplementedError("Please Implement this method")


class m_sim_mean(Model):
    def _initialize_model(self):
        self._initialize_forward()
    def _initialize_forward(self):
        self.forward_count = 0
        self.dW_img = np.zeros_like(self.W_img)
        self.dW_txt = np.zeros_like(self.W_txt)

    def backward(self, lr_img, lr_txt, batch_size):
        self.W_img = self.W_img - lr_img*self.dW_img/batch_size
        self.W_txt = self.W_txt - lr_txt*self.dW_txt/batch_size
        self._initialize_forward()
        return
        
    def compute_dS(self, x, y, S):
        norm_x = np.linalg.norm(x, 2)
        norm_y = np.linalg.norm(y,2)        
        dSx = ( y - x*S*norm_y/norm_x ) / (norm_x*norm_y)
        dSy = ( x - y*S*norm_x/norm_y ) / (norm_x*norm_y)
        return dSx.reshape(-1,1), dSy.reshape(-1,1)
        
    def train(self, images, texts, K,lr_img,lr_txt, batch_size=16, verbose=False):
        assert images.shape[0] == texts.shape[0], "Must have same number of images and texts"
        n_train = images.shape[0]        
        n_batches = n_train/batch_size
        for batch_id in range(n_batches):
            if verbose:
                fwrite('Mini-batch : %2d/%2d\r' % (batch_id, n_batches))
                sys.stdout.flush()
            begin = batch_id*batch_size
            end = (batch_id + 1)*batch_size
            for idx_p in range(begin, end):
                self.forward_count = 0
                im = images[idx_p]
                txt = texts[idx_p]
                x = np.dot(im, self.W_img)
                y = np.dot(txt, self.W_txt)
                t_txt = np.tile(txt.reshape(-1,1), self.n_hid).T
                t_im = np.tile(im.reshape(-1,1), self.n_hid).T
                s = cosine_similarity(x.reshape(1,-1),y.reshape(1,-1))[0][0]
                n = 0
                while self.forward_count<K:
                    #security                    
                    n += 1
                    if n >= 100:
                        break
                    idx_n = np.random.randint(n_train)
                    txt_n = texts[idx_n]
                    y_n = np.dot(txt_n, self.W_txt)
                    s_n = cosine_similarity(x.reshape(1,-1), y_n.reshape(1,-1))[0][0]
                    J = 0.5 + s_n - s
                    if J > 0.:
                        t_txt_n = np.tile(txt_n.reshape(-1,1), self.n_hid).T
                        self.update_grads(x, y, s, y_n, s_n, t_im, t_txt, t_txt_n)
            self.backward(lr_img,lr_txt, batch_size)
        if verbose: fwrite('\n')
    
    def update_grads(self, x, y, s, y_n, s_n, t_im, t_txt, t_txt_n):
        dSx, dSy = self.compute_dS(x,y,s)
        dSx_n, dSy_n = self.compute_dS(x,y_n,s_n)
        self.dW_img += np.transpose((dSx_n - dSx)*t_im)
        self.dW_txt += np.transpose(dSy_n*t_txt_n - dSy*t_txt)
        self.forward_count += 1
        
    def test(self, images, texts, epoch=0, saving_path = False):
        assert images.shape[0] == texts.shape[0], "Must have same number of images and texts"
        n_test = images.shape[0]    
        Xim_test_m = np.dot(images, self.W_img)
        Xtxt_test_m = np.dot(texts, self.W_txt)
        S = cosine_similarity(Xim_test_m, Xtxt_test_m)
        args = np.argsort(-S, axis=1)
        test_ranks = [(arg == idx).nonzero()[0][0] for idx, arg in enumerate(args)]
        d = S.diagonal()
        a = args[:,0]
        s = S[np.arange(n_test), a]
        
        if saving_path:
            plt.figure(figsize=(15,15))
            plt.title('Rank histogram epoch %d'% epoch)
            sns.distplot(test_ranks, label='Rank of the correct label (median = %d)' % np.median(test_ranks))
            plt.legend()
            plt.savefig(join(saving_path, 'rank_%d' %epoch))
        
            plt.figure(figsize=(15,15))
            plt.xlim([-0.1, 1.1])
            plt.title('Similarities epoch %d' % epoch)
            sns.distplot(d, label='Correct (im,txt)')
            sns.distplot(s, label='Best (im,txt)')
            plt.legend()
            plt.savefig(join(saving_path, 'sim_%d' % epoch))
        
        median_rank = np.median(test_ranks)
        return median_rank
   
class m_sim_mean_map(Model):
    def _initialize_model(self):
        self._initialize_forward()
    def _initialize_forward(self):
        self.forward_count = 0
        self.dW_img = np.zeros_like(self.W_img)
        self.dW_txt = np.zeros_like(self.W_txt)

    def backward(self, lr_img, lr_txt, batch_size):
        self.W_img = self.W_img - lr_img*self.dW_img/batch_size
        self.W_txt = self.W_txt - lr_txt*self.dW_txt/batch_size
        self._initialize_forward()
        return
        
    def compute_dS(self, x, y, S):
        norm_x = np.linalg.norm(x, 2)
        norm_y = np.linalg.norm(y,2)        
        dSx = ( y - x*S*norm_y/norm_x ) / (norm_x*norm_y)
        dSy = ( x - y*S*norm_x/norm_y ) / (norm_x*norm_y)
        return dSx.reshape(-1,1), dSy.reshape(-1,1)
        
    def train(self, images, texts, gamma,lr_img,lr_txt, batch_size=16, verbose = False):
        assert images.shape[0] == texts.shape[0], "Must have same number of images and texts"
        n_train = images.shape[0]        
        n_batches = n_train/batch_size
        indexes_n = np.arange(n_train)
        Ns = []
        for batch_id in range(n_batches):
            if verbose:
                fwrite('Mini-batch : %2d/%2d\r' % (batch_id, n_batches))
                sys.stdout.flush()
            begin = batch_id*batch_size
            end = (batch_id + 1)*batch_size
            for idx_p in range(begin, end):
                self.forward_count = 0
                im = images[idx_p]
                txt = texts[idx_p]
                x = np.dot(im, self.W_img)
                y = np.dot(txt, self.W_txt)
                t_txt = np.tile(txt.reshape(-1,1), self.n_hid).T
                t_im = np.tile(im.reshape(-1,1), self.n_hid).T
                s = cosine_similarity(x.reshape(1,-1),y.reshape(1,-1))[0][0]
                n = 0
                while n<gamma:
                    n+=1
                    idx_n = np.random.choice(indexes_n)
                    txt_n = texts[idx_n]
                    y_n = np.dot(txt_n, self.W_txt)
                    s_n = cosine_similarity(x.reshape(1,-1), y_n.reshape(1,-1))[0][0]
                    J = 1. + s_n - s
                    if J > 0.:
                        t_txt_n = np.tile(txt_n.reshape(-1,1), self.n_hid).T
                        self.update_grads(x, y, s, y_n, s_n, t_im, t_txt, t_txt_n)
                        break
                Ns.append(n)
            self.backward(lr_img,lr_txt, batch_size)
        if verbose:
            fwrite('\n')
        return np.mean(Ns)
    
    def update_grads(self, x, y, s, y_n, s_n, t_im, t_txt, t_txt_n):
        dSx, dSy = self.compute_dS(x,y,s)
        dSx_n, dSy_n = self.compute_dS(x,y_n,s_n)
        self.dW_img += np.transpose((dSx_n - dSx)*t_im)
        self.dW_txt += np.transpose(dSy_n*t_txt_n - dSy*t_txt)
        self.forward_count += 1
        
    def test(self, images, texts, epoch=0, saving_path = False):
        assert images.shape[0] == texts.shape[0], "Must have same number of images and texts"
        n_test = images.shape[0]    
        Xim_test_m = np.dot(images, self.W_img)
        Xtxt_test_m = np.dot(texts, self.W_txt)
        S = cosine_similarity(Xim_test_m, Xtxt_test_m)
        args = np.argsort(-S, axis=1)
        test_ranks = [(arg == idx).nonzero()[0][0] for idx, arg in enumerate(args)]
        d = S.diagonal()
        a = args[:,0]
        s = S[np.arange(n_test), a]
        
        if saving_path:
            plt.figure(figsize=(15,15))
            plt.title('Rank histogram epoch %d'% epoch)
            sns.distplot(test_ranks, label='Rank of the correct label (median = %d)' % np.median(test_ranks))
            plt.legend()
            plt.savefig(join(saving_path, 'rank_%d' %epoch))
        
            plt.figure(figsize=(15,15))
            plt.xlim([-0.1, 1.1])
            plt.title('Similarities epoch %d' % epoch)
            sns.distplot(d, label='Correct (im,txt)')
            sns.distplot(s, label='Best (im,txt)')
            plt.legend()
            plt.savefig(join(saving_path, 'sim_%d' % epoch))
        
        median_rank = np.median(test_ranks)
        return median_rank
        
        
        
class m_sim_mean_bidir(Model):
    def _initialize_model(self):
        self.n_iter = 0
        self.all_j1 = 0
        self.all_j2 = 0
        self._initialize_forward()
        
    def _initialize_forward(self):
        self._reset_counts()
        self.dW_img = np.zeros_like(self.W_img)
        self.dW_txt = np.zeros_like(self.W_txt)
        
    def _reset_counts(self):
        self.j1_count = 0
        self.j2_count = 0
        
    def _compute_dS(self, x, y, S):
        norm_x = np.linalg.norm(x, 2)
        norm_y = np.linalg.norm(y,2)        
        dSx = ( y - x*S*norm_y/norm_x ) / (norm_x*norm_y)
        dSy = ( x - y*S*norm_x/norm_y ) / (norm_x*norm_y)
        return dSx, dSy
        
    def _update_grads_J1(self,x,y,s,y_n,sy_n,t_im,t_txt,t_txt_n, dSx=None, dSy=None):
        if dSx is None:
            dSx, dSy = self._compute_dS(x,y,s)
        dSx_n, dSy_n = self._compute_dS(x,y_n,sy_n)
        dJ_txt = dSy_n*t_txt_n - dSy*t_txt
        dJ_im = (dSx_n - dSx) * t_im
        self.dW_img += dJ_im
        self.dW_txt += dJ_txt
        self.j1_count += 1
        self.all_j1 += 1
        return dSx, dSy
        
    def _update_grads_J2(self,x,y,s,x_n,sx_n,t_im,t_txt,t_im_n, dSx=None, dSy=None):
        if dSx is None:
            dSx, dSy = self._compute_dS(x,y,s)
        dSx_n, dSy_n = self._compute_dS(x_n,y,sx_n)
        dJ_txt = (dSy_n - dSy)*t_txt
        dJ_im = dSx_n*t_im_n - dSx*t_im
        self.dW_img += dJ_im
        self.dW_txt += dJ_txt
        self.j2_count += 1
        self.all_j2 += 1
        return dSx, dSy
    
    def backward(self, lr, batch_size):
        self.W_img = self.W_img - lr*self.dW_img/batch_size
        self.W_txt = self.W_txt - lr*self.dW_txt/batch_size
        self.n_iter += 1
        self._initialize_forward()
        return True
        
    def train(self, images, texts, K, lr, batch_size=16):
        assert images.shape[0] == texts.shape[0], "Must have same number of images and texts"
        n_train = images.shape[0]        
        n_batches = n_train/batch_size
        indexes_n = np.arange(n_train)
        for batch_id in range(n_batches):
            begin = batch_id*batch_size
            end = (batch_id + 1)*batch_size
            np.random.shuffle(indexes_n)
            for idx_p in range(begin, end):
                im = images[idx_p]
                txt = texts[idx_p]
                self._reset_counts()
                x = np.dot(im, self.W_img)
                y = np.dot(txt, self.W_txt)
                t_txt = np.tile(txt.reshape(-1,1), self.n_hid)
                t_im = np.tile(im.reshape(-1,1), self.n_hid)
                s = cosine_similarity(x.reshape(-1,1),y.reshape(-1,1))[0][0]
                for idx_n in indexes_n:
                    if idx_n == idx_p:
                        break
                    im_n = images[idx_n]
                    txt_n = texts[idx_n]
                    x_n = np.dot(im_n, self.W_img)
                    y_n = np.dot(txt_n, self.W_txt)
                    sx_n = cosine_similarity(x_n.reshape(-1,1), y.reshape(-1,1))[0][0]
                    sy_n = cosine_similarity(x.reshape(-1,1), y_n.reshape(-1,1))[0][0]
                    J1 = 0.5 + sy_n - s
                    J2 = 0.5 + sx_n - s
                    sys.stdout.flush()
                    if J1>0 and self.j1_count<K:
                        t_txt_n = np.tile(txt_n.reshape(-1,1), self.n_hid)
                        dSx, dSy = self._update_grads_J1(x,y,s,y_n,sy_n, t_im, t_txt, t_txt_n)
                    if J2>0 and self.j2_count<K:
                        t_im_n = np.tile(im_n.reshape(-1,1), self.n_hid)
                        dSx, dSy = self._update_grads_J2(x,y,s,x_n,sx_n,t_im,t_txt,t_im_n)
                    if self.j2_count >= K and self.j1_count >=K:
                        break
                self.backward(lr, batch_size)
            
    def test(self, images, texts, folder, epoch):
        assert images.shape[0] == texts.shape[0], "Must have same number of images and texts"
        n_test = images.shape[0]    
        Xim_test_m = np.dot(images, self.W_img)
        Xtxt_test_m = np.dot(texts, self.W_txt)
        S = cosine_similarity(Xim_test_m, Xtxt_test_m)
        args = np.argsort(-S, axis=1)
        test_ranks = [(arg == idx).nonzero()[0][0] for idx, arg in enumerate(args)]
        d = S.diagonal()
        a = args[:,0]
        s = S[np.arange(n_test), a]
        
        
        plt.figure(figsize=(15,15))
        plt.title('Rank histogram epoch %d'% epoch)
        sns.distplot(test_ranks, label='Rank of the correct label (median = %d)' % np.median(test_ranks))
        plt.legend()
        plt.savefig(join(folder, 'rank_%d' %epoch))
        
        
        plt.figure(figsize=(15,15))
        plt.xlim([-0.1, 1.1])
        plt.title('Similarities epoch %d' % epoch)
        sns.distplot(d, label='Correct (im,txt)')
        sns.distplot(s, label='Best (im,txt)')
        plt.legend()
        plt.savefig(join(folder, 'sims_%d' %epoch))
        
        median_rank = np.median(test_ranks)
        fwrite('\tTest median rank = %d\n' % median_rank)
        sys.stdout.flush()
        
class Multimodal(object):
    @classmethod
    def create(cls, n_img, n_txt, n_hid, l2=None, pooling=None, optim='sgd'):
        W_img_val = 0.01*np.random.randn(n_img, n_hid).astype(dtype)
        W_txt_val = 0.01*np.random.randn(n_txt, n_hid).astype(dtype)
        return cls(W_img_val, W_txt_val, l2, pooling, optim)
    
    @classmethod
    def load(cls,path):
#        W_img_val, W_txt_val, l2, pooling, optim = cPickle.load(open(path,'r'))
        params = cPickle.load(open(path,'r'))
        W_img_val, W_txt_val, l2 = params[:3]
        if len(params)<4:
            pooling = None
        if len(params)<5:
            optim = 'sgd'
        return cls(W_img_val, W_txt_val, l2, pooling, optim)
        
    def save(self, path):
        to_save = [self.W_img.get_value(), self.W_txt.get_value(), self.l2, self.pooling, self.optim]
        cPickle.dump(to_save, open(path,'w'))
        return None
        
    def __init__(self,W_img_val, W_txt_val, l2, pooling, optim):
        self.optim = optim
        self.pooling = pooling
        self.l2 = l2
        self.W_img = theano.shared(W_img_val, name='W_img')
        self.W_txt = theano.shared(W_txt_val, name='W_txt')
        self.params = [self.W_img, self.W_txt]
        
        self.image = T.vector('image')
        self.emb_image = T.dot(self.image, self.W_img)
        
        self.text_p = T.matrix('text_p')
        self.emb_text_p = T.dot(self.text_p, self.W_txt)
        
        self.text_n = T.tensor3('text_n')
        self.emb_text_n = T.dot(self.text_n, self.W_txt)
        
        
        def sim_2(x, y):
            x_r = x / (epsilon + x.norm(2))
            y_r = y / (epsilon + y.norm(2, axis=0))
            return T.dot(x_r, y_r)
        
        def sim_3(x,y):
            x_r = x /  x.norm(2)
            norms = y.norm(2, axis=-1)
            norms = T.reshape(norms, (norms.shape[0], norms.shape[1], 1))
            y_r = y / (epsilon + norms)
            return T.dot(y_r,x_r)
        def sim_3_mean(x,y):
            x_r = x /  x.norm(2)
            norms = y.norm(2, axis=-1)
            norms = T.reshape(norms, (norms.shape[0], 1))
            y_r = y / (epsilon + norms)
            return T.dot(y_r,x_r)
            
        if pooling == 'mean':
            self.sim_p = sim_2(self.emb_image, T.mean(self.emb_text_p, axis=1))
            self.sim_n = sim_3_mean(self.emb_image, T.mean(self.emb_text_n, axis=1))
        if pooling=='softmax':
            self.cos_p = sim_2(self.emb_image, self.emb_text_p.T)
            self.cos_n = sim_3(self.emb_image, self.emb_text_n)
            self.pre_sim_p = T.nnet.softmax(self.cos_p)
            self.pre_sim_n = T.nnet.softmax(self.cos_n)
            self.sim_p = T.max(self.pre_sim_p, axis=1)
            self.sim_n = T.max(self.pre_sim_n, axis=1)
        else:
            self.cos_p = sim_2(self.emb_image, self.emb_text_p.T)
            self.cos_n = sim_3(self.emb_image, self.emb_text_n)
            self.sim_p = T.max(self.cos_p)
            self.sim_n = T.max(self.cos_n, axis=1)
        self.maximum = T.maximum(0, 0.5 + self.sim_n - self.sim_p)
        self.cost = T.sum(self.maximum)
        
        if self.l2:
            for p in self.params:
                self.cost += (T.sum(p**2))
#        
        self.lr = T.scalar('lr')
        self.grads = T.grad(self.cost, self.params)
        self.updates = []
        if self.optim == 'sgd':
            self.updates = [(p, p-self.lr*g) for p,g in zip(self.params, self.grads)]
            
        elif self.optim == 'rmsprop':
            self.rms = [theano.shared(p.get_value()*0., name='rms_%s'%p) for p in self.params]
            for p, r, g in zip(self.params, self.rms, self.grads):
                new_r = 0.9*r + 0.1* g**2
                self.updates.append((r, new_r))
                self.updates.append((p, p-g*self.lr/(new_r+epsilon)))
        self._train = theano.function(inputs=[self.image, 
                                              self.text_p, 
                                              self.text_n, 
                                              self.lr],
                                      outputs = self.cost, 
                                      updates = self.updates,
                                      allow_input_downcast=True)
#        
#        # Test part
        self._test = theano.function(inputs=[self.image, self.text_p],
                                     outputs = self.sim_p,
                                     allow_input_downcast=True)
#
#
#    
    def train(self, image, text_p, xtxt_n, lr):
        lengths = [x.shape[0] for x in xtxt_n]
        m = max(lengths)
        text_n = np.array([np.pad(x, ((0,m-x.shape[0]), 
                                      (0,0)), mode='mean') for x in xtxt_n])
        return self._train(image, text_p, text_n, lr)
    
    def test(self, image, bag_of_vectors):
        return self._test(image, bag_of_vectors)