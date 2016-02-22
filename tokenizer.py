# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:22:55 2015

@author: hedibenyounes
"""
from string import punctuation
import re

word_breakers = re.compile(r'[\s{}]+'.format(re.escape(punctuation + ' ')))

sentence_breakers = re.compile(r'[{}]+'.format(re.escape('.!?\n')))
def sent_tokenize(raw_text):
    """
    Tokenize raw_text into multiple sentences
    """
    return [s.lower().strip() for s in sentence_breakers.split(raw_text)]
    
def word_tokenize(raw_sent):
    """
    Tokenizes raw_sent, and breaks it into tokens
    """
    return [w.strip() for w in word_breakers.split(raw_sent.replace(u'\xa0', ' ')) if len(w)>0]
    
if __name__ == "__main__":
    
    raw_text = "Bonjour ! Je m'appelle Philippe et j'aime beaucoup " + \
    "l'informatique. Même s'il est vrai que je ne suis pas encore très" + \
    " fort (et c'est un euphémisme ...), je suis animé d'une passion " + \
    "immense \nN'est-ce pas là un domaine porteur ? Et c'est sans parler de" + \
    """ ce que l'on appelle communément le "machine learning" ... =)""" +\
    "Et d'ailleurs, tout ceci était écrit dans l'article 53-432"
    
    for sent in sent_tokenize(raw_text):
        print word_tokenize(sent)
        print '-'*20