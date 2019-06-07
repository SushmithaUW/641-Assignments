import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import sys

if __name__ == "__main__":
    
    p = pd.read_csv(sys.argv[1], sep ='\n', names=['sentiment'])
    p['label'] = 'positive'
    
    
    n = pd.read_csv(sys.argv[2], sep ='\n', names=['sentiment'])
    n['label'] = 'negative'
    
    
    pv = pd.read_csv(sys.argv[3], sep ='\n', names=['sentiment'])
    pv['label'] = 'positive'
    
    
    nv = pd.read_csv(sys.argv[4], sep ='\n', names=['sentiment'])
    nv['label'] = 'negative'
    
    pt = pd.read_csv(sys.argv[5], sep ='\n', names=['sentiment'])
    pt['label'] = 'positive'
    
    
    nt = pd.read_csv(sys.argv[6], sep ='\n', names=['sentiment'])
    nt['label'] = 'negative'
    
    train = p.append(n)
    train = shuffle(train)
    
    
    val = pv.append(nv)
    val = shuffle(val)
    
    
    test = pt.append(nt)
    test = shuffle(test)
    
    w = CountVectorizer(ngram_range=(1,2),stop_words = None, tokenizer = None, preprocessor = None)
    
    X_train = w.fit_transform(train.sentiment)
    X_val = w.transform(val.sentiment)
    X_test = w.transform(test.sentiment)
    
    
    MNB = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
    MNB.fit(X_train, train.label)
    
    
    pred_val = MNB.predict(X_val)
    #print (np.mean(pred_val==val.label))
    
    pred_test = MNB.predict(X_test)
    print (np.mean(pred_test==test.label))
