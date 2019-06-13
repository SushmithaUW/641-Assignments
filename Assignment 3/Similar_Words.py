# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:43:18 2019

@author: Sushmitha Suresh
"""

import csv
import pandas as pd
import re
import gensim
from gensim.models import Word2Vec
import nltk
from sklearn.utils import shuffle
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load('mymodel.kv', mmap='r')
sim_words_good = word_vectors.wv.most_similar('good',topn=20)  
sim_words_bad = word_vectors.wv.most_similar('bad',topn=20)

print(sim_words_good)
print(sim_words_bad)