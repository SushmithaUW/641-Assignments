import csv
import pandas as pd
import re
import gensim
from gensim.models import Word2Vec
import nltk
from sklearn.utils import shuffle
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

p = pd.read_csv('Tokenized_with_stopword_pos.csv', sep ='\n', names=['sentiment'])
n = pd.read_csv('Tokenized_with_stopword_neg.csv', sep ='\n', names=['sentiment'])
p['label'] = 'positive'
n['label'] = 'negative'

p = shuffle (p.append(n))

for i in range(len(p.sentiment.values)):
    p.sentiment.values[i] = (p.sentiment.values[i].replace('\'', '').replace('[','').replace(',','').replace(']','')).split(' ')
    p.sentiment.values[i].remove('.')

    
model = Word2Vec(p.sentiment.values, min_count=1, window =150, workers=5)
model.save('mymodel.kv')
