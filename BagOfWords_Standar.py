#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:48:55 2018

@author: andreshita
"""
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans 
#------------------------------------------------------------------------------

#IMPORT DATA
#------------------------------------------------------------------------------
with open('spanish.txt', 'rt',encoding='utf-8') as Stop_words:
     vStop_words= []# List of stop or ignore words 
     for row in Stop_words:
         row = row.replace('\n', '')   
         vStop_words.append(row)
#------------------------------------------------------------------------------
with open('Anything_Else.txt', 'rt',encoding='utf-8') as File_Data:
    vAnythingElse = []   
    for row in File_Data:
         vAnythingElse.append(row)         


#------------------------------------------------------------------------------

# FUNCTIONS
#------------------------------------------------------------------------------
def fn_tokenize_sentences(sentence): 
    """
    Extract the words from a text, 
    then separate each word like a single object and finally makes words lower case
    """
    vWords = re.sub("[^\S]"," ",sentence).split() #nltk.word_tokenize(sentence)
    vWordsToken = [w.lower() for w in vWords]
    return(vWordsToken) 
#------------------------------------------------------------------------------
def fn_stop_words_clean(vListOfWords):
    """
    Wait for a list of words to clean all the "stopwords" in it
    """
    vCleanedList= []
    for word in vListOfWords:
        if word not in vStop_words:
           vCleanedList.append(word)
    return(vCleanedList)
#------------------------------------------------------------------------------    
def fn_counting_words(_vIdTokenphrase):
    """
    How many words exists in the database AE and how many are unique
    """
    vTotalWords = []
    for row in _vIdTokenphrase:
        for word in row[1]:
            vTotalWords.append(word)
    vLenghtTotalWords = len(vTotalWords)
    vUniqueWords = len(set(vTotalWords))
    return(vLenghtTotalWords,vUniqueWords)    
#------------------------------------------------------------------------------
def fn_fix_the_data():
    """
    It takes the AE phrases like income. Then it apply the fn_tokenize_sentences 
    and the fn_stop_words_clean functions to obtain as result a list of each
    tokenized AE phrase and its respective ID.
    """
    vIdTokenPhrase = []        
    for i in range(len(vAnythingElse)):        
        vWordTokenized= fn_tokenize_sentences(vAnythingElse[i])
        vWordWhitoutStopWords = fn_stop_words_clean(vWordTokenized)    
        vIdTokenPhrase.append([i,vWordWhitoutStopWords])
    return(vIdTokenPhrase)
#------------------------------------------------------------------------------
def fn_training_the_model():
    """
    The word2vec tool takes a text corpus as input and produces the word vectors as output. 
    It first constructs a vocabulary from the training text data and then learns vector representation of words. 
    Resource: https://code.google.com/archive/p/word2vec/
   
    """
    vIdTokenPhrase_ = fn_fix_the_data()
    vDataTrain = [] # Define training data/// Anything Else data set
    for id_prhase in vIdTokenPhrase_:
            vDataTrain.append(id_prhase[1])
    vModel = Word2Vec(vDataTrain, min_count=1,size=150,window=5,workers=4) # train model
    vModel.save('model_word2vect_ae.bin') # save model
    
#------------------------------------------------------------------------------
def fn_load_the_model():
    """ load the model: Word2vect """
    vLoadModel = Word2Vec.load('model_word2vect_ae.bin') 
    print("\nFeature's Model: ",vLoadModel)
    print('\n')
    return(vLoadModel)

#------------------------------------------------------------------------------


#PROCESS


#------------------------------------------------------------------------------

# Initial data parameters visualization

#------------------------------------------------------------------------------
print('\nAnything Else Phrases:',len(vAnythingElse)) # Lenght of AE phrases 
vIdTokenPhrase_ = fn_fix_the_data()
vLenghtTotalWords_ , vUniqueWords_ = fn_counting_words(vIdTokenPhrase_)
print('Total words:',vLenghtTotalWords_) 
vPercentageUniqueWords = ((vUniqueWords_ * 100) / vLenghtTotalWords_)
print('Unique words:',vUniqueWords_, '(' , "%.2f" % vPercentageUniqueWords ,'%)') 


#fn_training_the_model() 
vModelBOW = fn_load_the_model()
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#Convert each AE phrase in a vector by word:
vArrayWord2vec = []
for element in vIdTokenPhrase_:
    vAE_Sentence = element[1]
    vVectors= []
    vVectors.clear()
    for word in vAE_Sentence:
        vToVector = vModelBOW[word]  # get a vector for each word
        vVectors.append(vToVector)
    vArrayWord2vec.append([element[0],vVectors])


# Convert all vectors in just one ; sum of vectors
vArraySumOfVectors = []
for ae_vector in vArrayWord2vec:
    vSum_Vec = np.zeros((150,), dtype='float32')
    for i in range(len(ae_vector[1])):
        vSum_Vec  = vSum_Vec + ae_vector[1][i]
    vArraySumOfVectors.append([ae_vector[0],vSum_Vec]) # Id , vect_sum ===> AE phrases <=====
    
#------------------------------------------------------------------------------
  
# K-means model :Cluster

vXData = [] # Data
for row in vArraySumOfVectors: 
    vXData.append(row[1])

vClusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = vClusterNum, n_init = 12)

k_means.fit(vXData)

vLabels = k_means.labels_

print('\nNumber of phrases labeled:',len(vLabels))
print('Labels:', set(vLabels))
   

#------------------------------------------------------------------------------
# Classification of clusters

vCluster_0 = []
vCluster_1 = []
vCluster_2 = []
vCluster_3 = []
for index, label in enumerate(vLabels):
    if label == 0:
       vCluster_0.append(index)    
    if label == 1:
       vCluster_1.append(index) 
    if label == 2:
       vCluster_2.append(index)
    if label == 3:
       vCluster_3.append(index) 

print('\nClusters size')
print('Cluster 0: ', len(vCluster_0))
print('Cluster 1: ', len(vCluster_1))
print('Cluster 2: ', len(vCluster_2))
print('Cluster 3: ', len(vCluster_3))    

#------------------------------------------------------------------------------
# Recovery of sentences
vBag_0 = []
vBag_1 = []
vBag_2 = []
vBag_3 = []

for index, phrase in enumerate(vAnythingElse):
    if index in vCluster_0: 
       vBag_0.append(phrase)
    
    if index in vCluster_1: 
       vBag_1.append(phrase)
    
    if index in vCluster_2: 
       vBag_2.append(phrase)
    
    if index in vCluster_3: 
       vBag_3.append(phrase)
            
            
print('\nBag 0 has ', len(vBag_0), 'elements, for example:') 
print('')   
for i in range(20):
    print(vBag_0[i])
print('\n\n') 

print('Bag 1 has ', len(vBag_1), 'elements, for example:') 
print('')   
for i in range(20):
    print(vBag_1[i])
print('\n\n') 

print('Bag 2 has ', len(vBag_2), 'elements, for example:') 
print('')   
for i in range(20):
    print(vBag_2[i])
print('\n\n') 

print('Bag 3 has ', len(vBag_3), 'elements, for example:') 
print('')   
for i in range(20):
    print(vBag_3[i])
print('\n\n') 



#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
# Save into a excel file

#df0 = pd.DataFrame({'Bag 0':vBag_0})    
#df1 = pd.DataFrame({'Bag 1':vBag_1})
#df2 = pd.DataFrame({'Bag 2':vBag_2})
#df3 = pd.DataFrame({'Bag 3':vBag_3})   
#    
#    
#writer = pd.ExcelWriter('Anything_Else_Bag_of_words.xlsx')
#df0.to_excel(writer,'Bag 0')
#df1.to_excel(writer,'Bag 1')
#df2.to_excel(writer,'Bag 2')
#df3.to_excel(writer,'Bag 3')
#writer.save()   
#    
     
    
    
    
    
    
    
    
    
    
    
    
    