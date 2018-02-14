
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import os
import sys
# all words
words = ['dog','four','left','off','seven','three','wow','bed','down','go','marvin','on','sheila','tree','yes','bird','eight','happy','nine','one','six','two','zero','cat','five','house','no','right','stop','up','silence']
# reduced label indices
results = ['unknown','left', 'off' ,'down', 'go', 'on', 'yes', 'no' ,'right' ,'stop' ,'up' ,'silence']

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from random import randint


# In[ ]:


def getAudioFeatures(audiofilepath,  features = 'mfcc', augment=False ):
    
    try:
        (rate,sig) = wav.read(audiofilepath)
    
        if augment:
        # random pitch change +-3  and roll up to 2000 ( 12 % shift) 
            sig = pitch_n_roll(sig, rate, randint(-3, 3), randint(1, 2000) )
        
        
        if features == 'mfcc':
            return mfcc(sig,rate) / 45  # 45 seems to be the average +- of the data, normalize to fit the tanh activation range
        if features == 'logfbank':
            return logfbank(sig,rate)

    except:
        print("failed to convert " + audiofilepath)
        
        
        
        

def getBatch(index_df, batchsize, index_list, reduced_labels=False, audio_feat_type = 'mfcc' ):
    
    # reduced labels returns only samples with required labels
    
    trainlabels = [2, 3, 8, 9, 11, 14, 26, 27, 28, 29, 30]
    
    y_batch = []
    X_batch = []
    
    for f in range(batchsize):
        
        serie = index_df.loc[random.choice(index_list)]
        
        
        
        if not reduced_labels:
            y_batch.append(results.index(serie[0])) # use the original word list index
            X_batch.append( padding(getAudioFeatures(serie[1], audio_feat_type))) 
            
            
        if reduced_labels:
            if words.index(serie[0]) in trainlabels:
                
                X_batch.append( padding(getAudioFeatures(serie[1], audio_feat_type)))
                y_batch.append(results.index(serie[0])) # use the original word list index

             
        #print(padding(getAudioFeatures(serie[1])))
        
        
        
    return  X_batch, y_batch
                  
        
def padding(numpy_array, pad_type = "uniform",target_length = 99):
    """ add either gaussian noise or 0 arrays to the audio file until it matches 99 lenght """
    if(numpy_array.shape[0] == target_length):
        return numpy_array
    
    else:
        
        missing = target_length - numpy_array.shape[0]
        # fill in the gaps with 
        #for element in range(target_length - numpy_array.shape[0]):
            # add uniform random (0,1)  may be suboptimal
            
        # too short
        if missing > 0:
            if pad_type == "uniform":
                return np.vstack((numpy_array,np.random.rand(missing,13)))
            else:
                return np.vstack((numpy_array,np.zeros(missing,13)))
            
        # too long
        else:
            missing =  numpy_array.shape[0] -target_length 
            deletion = []
            
            for f in range(missing):
                deletion.append(numpy_array.shape[0] - f )
            return np.delete(numpy_array,deletion,0)
        
    
# prefetch data to memory
from tqdm import tqdm_notebook

def preload_data(index_list, index_df , audio_feat_type = 'mfcc', augment =False, n_augment = 4 ):
    dictionary = {}
    
    running_index = 0
    
    
    for element in index_list:
        
        if augment:
            serie = index_df.loc[element]
            
            for aug in range(n_augment):
                dictionary[running_index] = (results.index(serie[0]) , padding(getAudioFeatures(serie[1], audio_feat_type, augment)) )
                running_index +=1
        else:
            serie = index_df.loc[element]
            dictionary[element] = (results.index(serie[0]) , padding(getAudioFeatures(serie[1], audio_feat_type, augment)) )
    
    return pd.DataFrame.from_dict(dictionary, orient='index')





def batch_from_prefetch_samples(dataframe, batchsize):
    y_batch = []
    X_batch = []
    
    serie = dataframe.sample(n = batchsize, replace = True )
    
    try:
        y_batch = serie[0].tolist()
        X_batch = serie[1].tolist()
    except:
        # during reloading the headers sometimes change to string
        y_batch = serie['0'].tolist()
        X_batch = serie['1'].tolist()

    return  X_batch, y_batch

def fulldata_prefetch_samples_inorder(dataframe):
    """ Returns the data in order,  for final prediction """

    return dataframe[1].tolist(), dataframe[0].tolist()

# load the data 
import random

def split_val_data(ratio_train):
    """returns 2 list of indices"""
    
    train_idx = []
    val_idx = []

    for i in range(len(index_df)):
        if random.uniform(0, 1) <= ratio_train:
            train_idx.append(i)
        else:
            val_idx.append(i)
    
    return train_idx, val_idx


def augment_batch(X_batch, shift_max, noise_max = 1, add_shift = True, add_noise = True):
    """ make some noise, shift the data around """
    target_length = X_batch[0].shape[0]
    
    X_batch_out = []
    
    for X_elem in X_batch:
        
        if add_shift:
            shift = random.randint(0,shift_max)
            #print(shift)
        else:
            shift = 0
        deletion = []
        end = 0
        start = 1
        
        # cut randomly in the front and back of the columns
        for f in range(shift):
            if random.uniform(0.1, 1.1) > 0.55:
                deletion.append(  X_elem.shape[0] - (end+1) )
                
                #X_elem = np.delete(X_elem,(X_elem.shape[0] - (end+1),0)

                end +=1
            else:
                deletion.append(start)
                #X_elem = np.delete(X_elem,(start),0)

                start +=1

        X_elem = np.delete(X_elem,deletion,0)
        # randomly add the missing columns to the end or beginning of the dataset
        
        missing = target_length - X_elem.shape[0]
        if missing > 0 :
            if random.uniform(0, 1) > 0.5:
                # add to the front
                X_elem = np.vstack((X_elem,np.random.rand(missing,13)))
            
            else:
                # add to the back
                X_elem = np.vstack((np.random.rand(missing,13), X_elem))
            
            if add_noise:
                noise = np.random.normal(0,noise_max,(99*13)).reshape((99, 13))
            #X_batch_out.append(np.add(noise,X_elem))
    
        else:
            missing = X_elem.shape[0] -target_length 
            delete= []
            for element in range(missing):
                delete.append(X_elem.shape[0] - (element) )
            #print("element too long,removing one ")
            X_elem = np.delete(X_elem,delete,0)
           
            if add_noise:
                    noise = np.random.normal(0,noise_max,(99*13)).reshape((99, 13))
                    
        try:
            X_batch_out.append(np.add(noise,X_elem))
        except:
            X_elem = np.delete(X_elem,X_elem.shape[0]-1,0)
            X_batch_out.append(np.add(noise,X_elem))
    
    return X_batch_out
        #
        #X_out = np.add(X_elem,noise)
    
    

