# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:55:46 2019

@author: Sushmitha Suresh
"""


import numpy as np
import re
from nltk.corpus import stopwords
from random import randrange
import sys



if __name__ == "__main__":
    
 ##Assumed that the folder path does not contain spaces  
 
    input_path = sys.argv[1] 
    
##In case the folder path that you give contains spaces, kindly uncomment and run the following line:
    
   # input_path = sys.argv[len(sys.argv)-1]

     
    regex = r"\[P\]([\w\s]+)\[\/P\]"
    en_stops = set(stopwords.words('english'))
    
    t = []
    t1 = []
    
    def simple_tokenize(s):
        return re.findall(r"[\w']+|[.]",s.lower())
    
    def train_test_validate_split(df):
        df_copy = df
        train = list()
        train_size = 0.80 * len(df)
    
        while len(train) < train_size:
            index = randrange(len(df_copy))
            train.append(df_copy.pop(index))
        
        test_copy = df_copy
        test = list()
        test_size = 0.50 * len(df_copy)
    
        while len(test) < test_size:
            index = randrange(len(test_copy))
            test.append(test_copy.pop(index))
        
       
        return  train, test, test_copy
    
        
  ##Assumed that the filesnames will be kept as such. 
    
    if (input_path.find("pos")>0):
        input_path = open("pos.txt")
        for line in input_path:
            t.append(simple_tokenize(line.replace("'","")))
        np.savetxt("Tokenized_with_stopword_pos.csv",t, delimiter = ",", fmt="%s" )
    
        t1 = t
        for i in range(len(t1)):
            for item in t1[i]:
                if item in en_stops:
                    t1[i].remove(item)
        np.savetxt("Tokenized_without_stopword_pos.csv",t1, delimiter = ",", fmt="%s" )
    

        train_list,test_list,val_list = train_test_validate_split(t)
        train_list_no_stopword,test_list_no_stopword,val_list_no_stopword = train_test_validate_split(t1)
    
        np.savetxt("train_pos.csv", train_list, delimiter=",", fmt='%s')
        np.savetxt("val_pos.csv", val_list, delimiter=",", fmt='%s')
        np.savetxt("test_pos.csv", test_list, delimiter=",", fmt='%s')

        np.savetxt("train_no_stopword_pos.csv", train_list_no_stopword,delimiter=",", fmt='%s')
        np.savetxt("val_no_stopword_pos.csv", val_list_no_stopword, delimiter=",", fmt='%s')
        np.savetxt("test_no_stopword_pos.csv", test_list_no_stopword, delimiter=",", fmt='%s')  
    
    else:
        input_path = open("neg.txt")
        for line in input_path:
            t.append(simple_tokenize(line.replace("'","")))
        np.savetxt("Tokenized_with_stopword_neg.csv",t, delimiter = ",", fmt="%s" )
    
        t1 = t
        for i in range(len(t1)):
            for item in t1[i]:
                if item in en_stops:
                    t1[i].remove(item)
        np.savetxt("Tokenized_without_stopword_neg.csv",t1, delimiter = ",", fmt="%s" )
    

        train_list,test_list,val_list = train_test_validate_split(t)
        train_list_no_stopword,test_list_no_stopword,val_list_no_stopword = train_test_validate_split(t1)
    
        np.savetxt("train_neg.csv", train_list, delimiter=",", fmt='%s')
        np.savetxt("val_neg.csv", val_list, delimiter=",", fmt='%s')
        np.savetxt("test_neg.csv", test_list, delimiter=",", fmt='%s')

        np.savetxt("train_no_stopword_neg.csv", train_list_no_stopword,delimiter=",", fmt='%s')
        np.savetxt("val_no_stopword_neg.csv", val_list_no_stopword, delimiter=",", fmt='%s')
        np.savetxt("test_no_stopword_neg.csv", test_list_no_stopword, delimiter=",", fmt='%s')
    
       
        
        
        
            
    
        
    

   
  
    
    
    
    
    