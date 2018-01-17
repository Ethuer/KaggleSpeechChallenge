
# coding: utf-8

# In[8]:

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


# In[9]:

#  transforming data to cepstra
# The power cepstrum of a signal is defined as the squared magnitude of the 
# inverse Fourier transform of the logarithm of the squared magnitude of the Fourier transform of a signal:

# scale it to +-1 for the tanh activation of the RNN


# rarely goes above 40


# In[11]:

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from random import randint

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


# In[12]:

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




# In[13]:

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
    
    


# In[14]:

labels_to_numbers = {}
for f in range(len(results)):
    labels_to_numbers[results[f]] = f


# In[99]:

# get the available files from the training data directory

location = '../data/train/audio/'

index_dict = {}
index = 0
alt_label = 'unknown'

for label in words:
    if label in redux:
        for f in listdir('%s%s/' %(location,label)):
        #print(label, " ",f)
            index_dict[index] = (label,'%s%s/%s'%(location,label,f))
            index +=1
    else:
        print(label)
        for f in listdir('%s%s/' %(location,label)):
            index_dict[index] = (alt_label,'%s%s/%s'%(location,label,f))
            index +=1

index_df = pd.DataFrame.from_dict(index_dict, orient='index')
#index_df.rename(index=str,columns={0:"label",1:"location"})


# In[18]:

index_df.head()


# In[2]:

print(index_df[0].unique())


# In[52]:

## in late stages, a combined dataframe of uncertain samples emerged, 
## the score on the leaderboard diverged from the predicted score, so a list of bad samples, even if they are 
## in the test set improved estimation

realDat = pd.read_csv('../real_data_unsure.csv')

realDat[0] = realDat['label'].apply(lambda x: x)
realDat[1] = realDat['name'].apply(lambda x: '../data/test/audio/%s' %(x))

realDat = realDat.drop('name',axis=1)
realDat = realDat.drop('label',axis=1)


real_idx = []
for i in range(len(realDat)):
    real_idx.append(i)

prefetched_real = preload_data(real_idx,realDat,augment=False)
X_mess, y_mess = batch_from_prefetch_samples(prefetched_real,len(real_idx))


# In[101]:

train_idx, val_idx = split_val_data(0.99) # split the training data into test and validation


# In[24]:

# load the test data as well

sample_sub = pd.read_csv('../sample_submission.csv')

sample_sub[0] = sample_sub['label'].apply(lambda x: 'unknown')
sample_sub[1] = sample_sub['fname'].apply(lambda x: '../data/test/audio/%s' %(x))

sample_sub = sample_sub.drop('fname',axis=1)
sample_sub = sample_sub.drop('label',axis=1)


index_sample = []
for i in range(len(sample_sub)):
    index_sample.append(i)


# In[ ]:




# In[102]:

# prefetch   this takes a lot of time,  to speed up data extraction I augmented on the static

get_ipython().magic(u'time prefetched_data = preload_data(train_idx, index_df ,augment=False)')


# In[103]:

# test the time
get_ipython().magic(u'time X_batch, y_batch = batch_from_prefetch_samples(prefetched_data,90)')


# In[27]:

#prefetched_data.to_csv('../data/prefetched_20xaug_train.csv', index=False)


# In[28]:

prefetched_real = preload_data(real_idx,realDat,augment=False)


# In[29]:

X_mess, y_mess = batch_from_prefetch_samples(prefetched_real,len(real_idx))


# In[30]:

get_ipython().magic(u'time prefetched_val = preload_data(val_idx, index_df ,augment=False)')


# 

# In[31]:

print("running")


# In[32]:

get_ipython().magic(u'time prefetched_test = preload_data(index_sample,sample_sub, augment=False)')


# In[133]:

#prefetched_val.to_csv('../data/prefetched_val.csv', index=False)


# In[153]:

# restore the preloaded data

#prefetched_data = pd.read_csv('../data/prefetched_4xaug_train.csv', dtype=np.float32)
#prefetched_val = pd.read_csv('../data/prefetched_val.csv', dtype=np.float32)


# In[55]:

#prefetched_val = prefetched_val.drop("Unnamed: 0", axis=1)
#prefetched_val = prefetched_val.rename(index=str,columns={'0':0,'1':1})


# In[56]:

#prefetched_data = prefetched_data.drop("Unnamed: 0", axis=1)
#prefetched_data = prefetched_data.rename(index=str,columns={'0':0,'1':1})


# In[22]:

#%time X_test, y_test = getBatch(index_df,len(val_idx),val_idx)
#X_subs, y_subs = getBatch(index_df,4000,val_idx,reduced_labels=True)


# In[33]:

get_ipython().magic(u'time X_batch, y_batch = batch_from_prefetch_samples(prefetched_data,100)')


# In[34]:

get_ipython().magic(u'time data,y_empty = fulldata_prefetch_samples_inorder(prefetched_test)')


# In[35]:

X_batch, y_batch = batch_from_prefetch_samples(prefetched_test,100)


# In[36]:

get_ipython().magic(u'time X_test,y_test = batch_from_prefetch_samples(prefetched_val,len(val_idx))')


# In[123]:

# build the RNN


# build tensorboard setup 
# needs a root_logdir folder. 
# saves tensorboard output there under timestamp
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logdir"
logdir = "{}/run-{}".format(root_logdir, now)


# In[124]:

tf.reset_default_graph()  

# autoencoder parameters for pretraining 

# I couldn't build a full conv2d network, but had to cut it down to fit memory
# To remove noise, a denoisying autoencoder is trained



an_inputs = 99*13
an_hidden1 = 800
an_hidden2 = 550  # codings
an_hiddencore = 200  # codings


# tying the weights of output to input.  speeds things up
an_hiddencoreout = an_hidden2
an_hidden3 = an_hidden1
an_outputs = an_inputs

a_learning_rate_initial = 0.00007  # empirically good enough.
a_decay_steps = 10000
a_decay_rate = 0.95

l2_reg = 0.0004

a_global_step = tf.Variable(0,trainable=False)

#  Autoencoder generation.  Can just run in same graph, mostly because importing is annoying

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None,99,13] )



training = tf.placeholder_with_default(False,shape=(),name='training')
Aenc__training = tf.placeholder_with_default(False,shape=(),name='training')


a_weights1_init = initializer([an_inputs, an_hidden1])
a_weights2_init = initializer([an_hidden1, an_hidden2])
a_weightscore_init = initializer([an_hidden2, an_hiddencore])

# weights for autoencoder,  linking 1&4 and 2&3  so I only need to train 1 and 2
a_weights1 = tf.Variable(a_weights1_init, dtype=tf.float32, name="a_weights1")
a_weights2 = tf.Variable(a_weights2_init, dtype=tf.float32, name="a_weights2")
a_weightscore = tf.Variable(a_weightscore_init, dtype=tf.float32, name="a_weightscore")

a_weightscoreout = tf.transpose(a_weightscore, name="a_weightscoreout") # tied weights
a_weights3 = tf.transpose(a_weights2, name="a_weights3")  # tied weights
a_weights4 = tf.transpose(a_weights1, name="a_weights4")  # tied weights

# biases for autoencoder,  not linked
a_biases1 = tf.Variable(tf.zeros(an_hidden1), name="a_biases1")
a_biases2 = tf.Variable(tf.zeros(an_hidden2), name="a_biases2")
a_biasescore = tf.Variable(tf.zeros(an_hiddencore), name="a_biasescore")
a_biasescoreout = tf.Variable(tf.zeros(an_hiddencoreout), name="a_biasescoreout")
a_biases3 = tf.Variable(tf.zeros(an_hidden3), name="a_biases3")
a_biases4 = tf.Variable(tf.zeros(an_outputs), name="a_biases4")

#X_flat = tf.contrib.layers.flatten(X)

with tf.name_scope("pretraining_Autoencoder"):
    X_flat = tf.contrib.layers.flatten(X)
    a_X_drop = tf.layers.dropout(X_flat,0.5,training=a_training)
    a_hidden1 = activation(tf.matmul(a_X_drop, a_weights1) + a_biases1)
    a_hidden2 = activation(tf.matmul(a_hidden1, a_weights2) + a_biases2)
    
    a_hiddencore = activation(tf.matmul(a_hidden2, a_weightscore) + a_biasescore)
    
    #a_hiddencoreout = activation(tf.matmul(a_stop, a_weightscoreout) + a_biasescoreout)
    a_hiddencoreout = activation(tf.matmul(a_hiddencore, a_weightscoreout) + a_biasescoreout)
    
    a_hidden3 = activation(tf.matmul(a_hiddencoreout, a_weights3) + a_biases3)
    a_outputs = tf.matmul(a_hidden3, a_weights4) + a_biases4

    a_outputs_reshaped = tf.reshape(a_outputs, shape=[-1,99,13])
    
    a_reconstruction_loss = tf.reduce_mean(tf.square(a_outputs - X_flat))
    a_reg_loss = regularizer(a_weights1) + regularizer(a_weights2)
    a_loss = a_reconstruction_loss + a_reg_loss

    a_learning_rate = tf.train.natural_exp_decay(a_learning_rate_initial,a_global_step,a_decay_steps,a_decay_rate)
    a_optimizer = tf.train.AdamOptimizer(a_learning_rate)
    a_training_op = a_optimizer.minimize(a_loss, global_step=a_global_step)


# In[125]:

# not the actual network


# In[126]:



n_steps = 13
n_inputs = 99
n_neurons = 300 
n_outputs = len(results)

initial_learning = 0.00015
decay_steps = 30000
decay_rate = 0.98

momentum = 0.995  # adam outperforms nesterov

# overfitting is an issue even with dropoit,  added regularizer additionally
scale =0.0014  # regularizer scale  0.0 switches them off


# In[127]:

# Neural net construction


y = tf.placeholder(tf.int32,[None])



# In[128]:

cells = [
        
        tf.contrib.rnn.GRUCell(num_units=n_neurons,activation=tf.nn.tanh), #relu outperforms elu but tanh matches the activation range
        
         
         tf.contrib.rnn.DropoutWrapper(
         tf.contrib.rnn.GRUCell(num_units=(n_neurons-10),activation=tf.nn.tanh,  # relus get > 88%
                               kernel_initializer = tf.contrib.layers.xavier_initializer())
          , input_keep_prob = 0.3 ),
    
        tf.contrib.rnn.DropoutWrapper(
         tf.contrib.rnn.GRUCell(num_units=(n_neurons-30),activation= tf.nn.relu, #tf.nn.tanh,  # relus get > 88%
                               kernel_initializer = tf.contrib.layers.xavier_initializer())
          , input_keep_prob = 0.3 ),
    

        tf.contrib.rnn.OutputProjectionWrapper(
            
            tf.contrib.rnn.DropoutWrapper(
         tf.contrib.rnn.GRUCell(num_units=(n_neurons-10),activation=tf.nn.relu, #tf.nn.tanh,  # relus get > 88%
                               kernel_initializer = tf.contrib.layers.xavier_initializer())
          , input_keep_prob = 0.3 ),
         
            output_size = n_outputs
        )

    
        ]
                                
        
with tf.name_scope("rnn"):
    
    # 2 layer GRUs work to 90% singleaccurary with some Augmentation on all training data,
    # needs a few more layers to generalize the unwanted labels to unknown
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)
    
    # both outputs ( states over times) and states ( final neuron states ) contain information, so using both of those. 
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    
    outputs_flat = tf.reshape(outputs, [-1, ( n_inputs*12)])
    


# In[129]:

# this convolutional NN fits the autoencoded data, it starts from a low res input
# the network alone, when properly trained reaches > 60% accuracy.

with tf.name_scope("Conv"):
    
    conv_learning_rate_init = 0.002
    conv_decay_steps = 10000
    conv_decay_rate = 0.95
    
    conv_global_step = tf.Variable(0, trainable=False)

    #conv_learning_rate = tf.train.exponential_decay(initial_learning,conv_global_step,conv_decay_steps,conv_decay_rate)
    conv_learning_rate = 0.002
    
    train_convNet = tf.placeholder_with_default(False,shape=(),name='training')

    a_output_stop_fromconv = tf.stop_gradient(a_outputs_reshaped)
    input_2d =  tf.expand_dims(a_output_stop_fromconv,3)
    
    
    
    normresponse_input = tf.nn.local_response_normalization(input_2d)
    
    low_res_input = tf.nn.max_pool(input_2d,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding="SAME")
    
    
    
    conv_layer = tf.layers.conv2d(low_res_input ,
                                  filters=32,
                                  kernel_size = [2,2],
                                  padding='SAME', 
                                  activation=tf.nn.tanh )
 
    normresponse_conv_1 = tf.nn.local_response_normalization(conv_layer)

    max_pool_in = tf.nn.max_pool(normresponse_conv_1, ksize=[1,2, 2,1], strides=[1,2, 2,1], padding="SAME",)
    
    normresponse = tf.nn.local_response_normalization(max_pool_in)
    
    conv_layer2 = tf.layers.conv2d(normresponse ,filters=32,kernel_size = [3,2],dilation_rate=2, padding='SAME')
    conv_layer_elu2 = tf.nn.tanh(conv_layer2)
    
    normresponse_2 = tf.nn.local_response_normalization(conv_layer_elu2)
    
    conv_layer3 = tf.layers.conv2d(normresponse_2 ,filters=32,kernel_size = [3,2],dilation_rate=2, padding='SAME')
    conv_layer_elu3 = tf.nn.tanh(conv_layer3) 
    
    normresponse_3 = tf.nn.local_response_normalization(conv_layer_elu3)
    
    
    pool_flat_in = tf.contrib.layers.flatten(normresponse_3)
    
    act_bn_norm = tf.layers.batch_normalization(pool_flat_in,training=train_convNet, momentum = momentum)
                             
    act_bn_norm_act = tf.nn.elu(act_bn_norm)
    
    act_elu = tf.layers.dense(act_bn_norm_act, units=n_neurons, activation=tf.nn.elu)
    
    act_drop = tf.layers.dropout(act_elu, training = train_convNet, rate=0.5 )
    
    act_elu_stage2 = tf.layers.dense(act_drop, units=n_neurons - 15, activation=tf.nn.elu)
    
    act_drop2 = tf.layers.dropout(act_elu_stage2, training = train_convNet, rate=0.5 )
    
    
    # outgoing layer, stops the gradient 
    conv_stop = tf.stop_gradient(act_drop2) # don't propagate the gradient back any further
    conv_outgoing = tf.layers.dense(conv_stop, units= n_neurons - 15 )
    
    # training and evaluation for the Convnet alone
    
    conv_logits = tf.layers.dense(act_drop2, n_outputs, name="conv_outputs_logits") 
    
    conv_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=conv_logits)
    conv_loss = tf.reduce_mean(conv_xentropy)
    conv_optimizer = tf.train.AdamOptimizer(learning_rate=conv_learning_rate)# Adamoptimizer implementing the nesterov step 
    conv_training_op = conv_optimizer.minimize(conv_loss)
    conv_correct = tf.nn.in_top_k(conv_logits, y, 1)
    conv_accuracy = tf.reduce_mean(tf.cast(conv_correct, tf.float32))
    
    
    
    


# In[132]:

with tf.name_scope("DenseFromEncoder"):
    encoder_stop = tf.stop_gradient(a_hiddencore)
    
    input_encoder = encoder_stop # using the central layer of the encoder
    
    first_encoder = tf.layers.dense(input_encoder, units=n_neurons, activation=tf.nn.elu)

    # now glue this into the core pipeline, and fix the autoencoder weights while training


# In[133]:

#states_concat = tf.concat(axis=1, values=states)

with tf.name_scope("parted_layer_output"):
    
    # outputs are a tensor of  batch,timepoints(99),cffs(13)
    # feeding this into a 3d convnet need to add channels1  tf.expand_dims(t, 1)
    output_2d = tf.expand_dims(outputs,3)
    low_res_RNN_out = tf.nn.avg_pool(output_2d,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding="SAME")
    # reduce resolution
    
    
    conv2d = tf.layers.conv2d(low_res_RNN_out,filters=20,kernel_size = [3,3],strides=[1,1], padding='SAME')
    convoutput_2d_act = tf.nn.elu(conv2d)
    
    normresponse_out1 = tf.nn.local_response_normalization(convoutput_2d_act)
    
    conv2d_2 = tf.layers.conv2d(normresponse_out1,filters=20,kernel_size = [3,3],strides=[1,1], padding='SAME')
    convoutput_2d_act_2 = tf.nn.elu(conv2d_2)
    
    
    normresponse_out1 = tf.nn.local_response_normalization(convoutput_2d_act_2)
    # 99 x 12   => 33 x6
    # 99 x 12  [3,6] => 33*2
    max_pool = tf.nn.max_pool(normresponse_out1, ksize=[1,3, 3,1], strides=[1,3, 3,1], padding="SAME")

    
    #normresponse_out2 = tf.nn.local_response_normalization(max_pool)
    
    #pool_flat = tf.reshape(conv2d, [-1, 33*6*  32])
    pool_flat = tf.contrib.layers.flatten(max_pool)
    
    dense_out = tf.layers.dense(pool_flat, (n_neurons), #activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name="supplementationDense_fullRNNoutput") # add a dense layer
    
    
with tf.name_scope("parted_layer_states"):   
    
    dense_state = tf.layers.dense(states, (n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name="supplementationDense_covering_states") # add a dense layer
    
    
with tf.name_scope("combine_states"):
    
    
    combinedstates = tf.concat([dense_state, dense_out,conv_outgoing], 1)

    #combinedstates = tf.add(dense_state, dense_out)
    
    

with tf.name_scope("supplement_layer"):
   
   
    dense2 = tf.layers.dense(combinedstates, (n_neurons * 2 ), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            
                             kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(scale),
                           name="supplementationDense2") # add a dense layer
    
    #batch normalization  
    # doesn't improve output applied before skip connections
    
    
    bn_norm_sup2 = tf.layers.batch_normalization(dense2,training=training, momentum = momentum)
                             
    bn_norm_act2 = tf.nn.elu(bn_norm_sup2) # activate the batch norm
    
    
     # combine with states here 
        
    drop2_ext = tf.concat([bn_norm_act2, states], 1)
    
    # addition
 
    
    bn_norm_sup_ext = tf.layers.batch_normalization(drop2_ext,training=training, momentum = momentum)
                             
    bn_norm_act_ext = tf.nn.elu(bn_norm_sup_ext)
    
    
    
    
    dense3 = tf.layers.dense(bn_norm_act_ext, (n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale),
                           name="supplementationDense3") # add a dense layer
    
    
    # add autoencoder data here
    
    dense_encoded = tf.concat([dense3, first_encoder], 1)
    
    
    bn_norm_enc_ext = tf.layers.batch_normalization(dense_encoded,training=training, momentum = momentum)
                             
    bn_norm_enc_ext = tf.nn.elu(bn_norm_enc_ext)
    
    dense3_encode = tf.layers.dense(bn_norm_enc_ext, (n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale),
                           name="supplementationDense3enc") # add a dense layer
    
    drop3 = tf.layers.dropout(dense3_encode, training = training, rate=0.3 )
    
    dense3_b = tf.layers.dense(drop3, (n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale),
                           name="supplementationDense3_b") # add a dense layer
    
    bn_norm_sup3 = tf.layers.batch_normalization(dense3_b,training=training, momentum = momentum)
                             
    bn_norm_act3 = tf.nn.elu(bn_norm_sup3) # activate the batch norm
    
    
    
     # combine with conv_part here

    drop3_ext = tf.concat([dense3, conv_outgoing], 1)
    
    #addition
    
    dense4 = tf.layers.dense(drop3_ext, (n_neurons - 10), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale),
                           name="supplementationDense4") # add a dense layer
    
    bn_norm_sup4 = tf.layers.batch_normalization(dense4,training=training, momentum = momentum)                  
    bn_norm_act4 = tf.nn.elu(bn_norm_sup4) # activate the batch norm
    
    
    drop4 = tf.layers.dropout(bn_norm_act4, training = training, rate=0.2 )
    
    # combine with output here
    
    dense4_ext = tf.concat([drop4, states], 1)
    
  
    dense5 = tf.layers.dense(dense4_ext, (n_neurons - 75), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale),
                           name="supplementationDense5") # add a dense layer
    
    
    drop5 = tf.layers.dropout(dense5, training = training, rate=0.5 )
    
    dense6 = tf.layers.dense(drop5, (n_neurons - 75), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale),
                           name="supplementationDense6") # add a dense layer
    
    

with tf.name_scope("logits"):

    logits_before_bn = tf.layers.dense(dense6, n_outputs, name="outputs_logits") #logits = tf.layers.dense(states, n_outputs)
    logits = tf.layers.batch_normalization(logits_before_bn,training=training, momentum = momentum)
    
    

#  for training of the RNN
with tf.name_scope("loss"):
    
    
    
    # sparse softmax, so I don't have to get the ohot
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    
    # regularization losses
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # ignore regularization  it's not helping
    loss = tf.reduce_mean(xentropy)
    #
    #  activate regularization here
    loss_aftereg = tf.add_n([loss]+ reg_loss)
    
    
    
with tf.name_scope("train"):
    
    
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(initial_learning,global_step,decay_steps,decay_rate)
    #learning_rate = tf.train.natural_exp_decay(initial_learning,global_step,decay_steps,decay_rate)
    
    #learning_rate = 0.00074
    optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate) # Adamoptimizer implementing the nesterov step 
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate) # recommended for RNN,  not optimal performance
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    training_op = optimizer.minimize(loss_aftereg, global_step=global_step)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
with tf.name_scope("logging"):
    
    accuracy_sum = tf.summary.scalar('accuracy',accuracy)
    loss_sum = tf.summary.scalar('loss',loss)
    
    summaries = tf.summary.merge_all()
    
    # writ to tensorboard
    train_writer = tf.summary.FileWriter(logdir + '/train', tf.get_default_graph())
    test_writer = tf.summary.FileWriter(logdir + '/test')
    
    
with tf.name_scope("predict"):
    prediction = tf.argmax(logits,1)
    



# In[134]:

init = tf.global_variables_initializer()


# In[135]:

saver = tf.train.Saver()
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


# In[ ]:

def multiclass_accuracy(x_lab, x_pred):
    classdict = {}
    all_list = []
    TP = 0.0
    total = 0.0
    
    for index in range(len(x_lab)):
        if not x_lab[index] in classdict:
            classdict[x_lab[index]] = [0.0,0.0]
        
    for index in range(len(x_lab)):
        
        if x_lab[index] == x_pred[index]:
            classdict[x_lab[index]][0] +=1
            
        if x_lab[index] != x_pred[index]:
            classdict[x_lab[index]][1] +=1
        classdict[x_lab[index]][1] +=1
    
    
    
    for key,vals in classdict.items():

        all_list.append(float(vals[0]/float(vals[1])))
        
    
    return np.mean(all_list)
                  


# In[137]:

# train autoencoder alone first on the unlabeled test data,  it needs no labels anyways
X_mess, y_mess = batch_from_prefetch_samples(prefetched_real,len(real_idx))
get_ipython().magic(u'time X_test,y_test = batch_from_prefetch_samples(prefetched_val,3500)')


# In[71]:

n_epochs = 25 # 33
batch_size = 320#250

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range( prefetched_test.shape[0]// (batch_size) ):
            X_batch, y_batch = batch_from_prefetch_samples(prefetched_test,batch_size)
            #X_batch = X_batch.tolist()
            
            n_batches = prefetched_test.shape[0] / (batch_size)
            
            print("\r{}%".format(100 * iteration // n_batches), end="")
                            
            sys.stdout.flush()

            sess.run(a_training_op, feed_dict={Aenc_training: True, X: X_batch})
        
        
        loss_train = a_reconstruction_loss.eval(feed_dict={X: X_batch})
        loss_test = a_reconstruction_loss.eval(feed_dict={X: X_mess})
        print("\r{}".format(epoch), "Train MSE:", loss_train, " test MSE:", loss_test)    
        
    saver.save(sess, "Autoencoder_checkpoints/FirstStep_Autoencoderonly_%s.ckpt" %epoch)


# In[87]:

# train the convolutional input

n_epochs = 100
batch_size = 946#  90#220#370 #250#180 # 90

with tf.Session() as sess:
    init.run()
    saver.restore(sess,'Autoencoder_checkpoints/FirstStep_Autoencoderonly_24.ckpt') 

    for epoch in range(n_epochs):
        for iteration in range( index_df.shape[0] // (batch_size) ):
            X_batch, y_batch = batch_from_prefetch_samples(prefetched_data,batch_size)
            
            
            sess.run([conv_training_op], feed_dict={train_convNet : True, X: X_batch, y: y_batch, })
            
            
        acc_train = accuracy.eval(feed_dict={X: X_test, y: y_test})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        
        if acc_test >0.6:
            save_path = saver.save(sess, "Conv_checkpoint/Intermed_conv_auto_%s.ckpt" %acc_test)
            
        print(epoch, " Conv Train accuracy:", acc_train " test" , acc_test)
         
    save_path = saver.save(sess, "Conv_checkpoint/Final_conv_auto.ckpt")


# In[138]:

n_epochs = 420
batch_size = 300



with tf.Session() as sess:
    init.run()
    count = 0
    
    # load the autoencoder to pre-process the data
    saver.restore(sess,'Conv_checkpoint/Final_conv_auto.ckpt') 
   
    
    for epoch in range(n_epochs):

        for iteration in range( index_df.shape[0] // (batch_size * 8) ):
            X_batch, y_batch = batch_from_prefetch_samples(prefetched_data,batch_size)
            X_batch_aug = augment_batch(X_batch= X_batch, noise_max=0.01, shift_max=8)
            sess.run([training_op,extra_update_ops], feed_dict={training: True, X: X_batch_aug, y: y_batch})
            
            if iteration % 200 == 0:
                step = epoch*iteration
                summary, _ = sess.run([summaries,extra_update_ops], feed_dict={X: X_test, y: y_test} )
                train_writer.add_summary(summary, step )
                
                
                summary, _ = sess.run([summaries,extra_update_ops], feed_dict={X: X_mess, y: y_mess})
                
                test_writer.add_summary(summary,step)
                
                

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
       
        
        step = epoch*iteration
        summary, _ = sess.run([summaries,extra_update_ops], feed_dict={X: X_test, y: y_test} )
        train_writer.add_summary(summary, step )     
        summary, _ = sess.run([summaries,extra_update_ops], feed_dict={X: X_mess, y: y_mess})    
        test_writer.add_summary(summary,step)
        
        
      
        #if count % 5 == 0:
        #    save_path = saver.save(sess, "RNN_4_concat_conv2d_ampdata_1001_savedbycount_%s_%s.ckpt" %(epoch,acc_mess) )
        #count +=1
            
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test}) # all words from directories
        test_pred = prediction.eval(feed_dict={X: X_test})
        multi_test = multiclass_accuracy(y_test,test_pred)
        
        if multi_test > 0.98:
            save_path = saver.save(sess, "RNN_4_c_concat_conv2d_ampdata_1101_c1_checkpoint_scorebased_%s.ckpt" %(multi_acc) )
            print("written savepoint %s ", %(multi_test))
        
        print(epoch, "Train accuracy:", round(acc_train,2), " - ", round(acc_test,2),"mediocre :", "ma test: ", round(multi_test,2))
    save_path = saver.save(sess, "Final_training_auto_conv_RNN.ckpt")


# In[ ]:

# predictions


# In[143]:


# all samples at once causes and OOM error... chunking time
def chunker(seq, size):
    return (data[pos:pos + size] for pos in range(0, len(data), size))


# In[158]:

with tf.Session() as sess:
    #saver = tf.train.import_meta_graph("RNN_3layer_drop.ckpt")
    
    saver.restore(sess, "Final_training_auto_conv_RNN.ckpt")

    reslist = []
    
    for group in chunker(data,1000):
        results = prediction.eval(feed_dict={X:group})
        for element in results.tolist():
            reslist.append(element)
            
        


# In[160]:

full_prediction_list = reslist
results = ['unknown','left', 'off' ,'down', 'go', 'on', 'yes', 'no' ,'right' ,'stop' ,'up' ,'silence']

sample_sub = pd.read_csv('../sample_submission.csv')

sample_sub['pred'] = full_prediction_list
sample_sub['label'] = sample_sub['pred'].apply(lambda x: results[x] )

sample_sub = sample_sub.drop(labels='pred', axis=1)
sample_sub.to_csv("../test_submission_RNN_predicted.csv", index=False)
sample_sub.head()


# 
