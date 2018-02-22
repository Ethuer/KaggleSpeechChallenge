
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import os
import sys


# In[ ]:




# In[81]:

class Multilayer_RNN():
    def __init__(self, an_inputs = 99*13,an_hidden1 = 800,
                 an_hidden2 = 550, an_hiddencore = 200,a_learning_rate_initial = 0.00007,
                 a_decay_steps = 10000,a_decay_rate = 0.95,
                 l2_reg = 0.0004,conv_learning_rate_init = 0.002,conv_decay_steps = 10000,
                 conv_decay_rate = 0.95,conv_learning_rate = 0.002,dropout = 0.5,
                 input_keep_prob = 0.3,n_steps = 13,n_inputs = 99,
                 n_neurons = 300 ,n_outputs = 7,
                 initial_learning = 0.00015,decay_steps = 30000,
                 decay_rate = 0.98,momentum = 0.995 ,scale =0.0014,
                logdir = 'Tensorboard_logdir'):
        
        
        
        tf.reset_default_graph()
        
        
        self.logdir = logdir
        
        # autoencoder layers
        self.an_inputs = an_inputs
        self.an_hidden1 = an_hidden1
        self.an_hidden2 = an_hiddencore
        self.an_hiddencore = an_hiddencore
        self.an_hiddencoreout = an_hiddencore
        self.an_hidden3 = an_hidden1
        self.an_outputs = an_inputs

        # autoencoder parameters
        
        self.a_learning_rate_initial = a_learning_rate_initial = 0.00007 
        self.a_decay_steps = a_decay_steps
        self.a_decay_rate = a_decay_rate

        self.l2_reg = l2_reg
        self.activation = tf.nn.elu
        self.regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        
        # Convolutional parameters
        self.conv_learning_rate_init = conv_learning_rate_init
        self.conv_decay_steps = conv_decay_rate
        self.conv_decay_rate = conv_decay_rate
        self.conv_learning_rate = conv_learning_rate
        self.dropout = dropout
        
        # RNN parameters
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.initial_learning = initial_learning
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.scale = scale
        self.input_keep_prob = input_keep_prob
        
        
        
        
        self.X = tf.placeholder(tf.float32, shape=[None,99,13] )
        self.y = tf.placeholder(tf.int32,[None])
        
        self.training = tf.placeholder_with_default(False,shape=(),name='training')
        self.a_training = tf.placeholder_with_default(False,shape=(),name='AENCtraining')
        self.train_convNet = tf.placeholder_with_default(False,shape=(),name='CONVtraining')
        
        
        
        
        
    def _build(self):
        
        
        
        # Autoencoder first,  tying input output weights, since tf layers don't really support this, manipulate the weights directly
        
        
        
        self.a_global_step = tf.Variable(0,trainable=False)

        
        with tf.name_scope("Autoencoder"):
            #  Autoencoder generation.  Can just run in same graph, mostly because importing is annoying

            self.a_weights1_init = self.initializer([self.an_inputs, self.an_hidden1])
            self.a_weights2_init = self.initializer([self.an_hidden1, self.an_hidden2])
            self.a_weightscore_init = self.initializer([self.an_hidden2, self.an_hiddencore])

            # weights for autoencoder,  linking 1&4 and 2&3  so I only need to train 1 and 2
            self.a_weights1 = tf.Variable(self.a_weights1_init, dtype=tf.float32, name="a_weights1")
            self.a_weights2 = tf.Variable(self.a_weights2_init, dtype=tf.float32, name="a_weights2")
            self.a_weightscore = tf.Variable(self.a_weightscore_init, dtype=tf.float32, name="a_weightscore")

            self.a_weightscoreout = tf.transpose(self.a_weightscore, name="a_weightscoreout") # tied weights
            self.a_weights3 = tf.transpose(self.a_weights2, name="a_weights3")  # tied weights
            self.a_weights4 = tf.transpose(self.a_weights1, name="a_weights4")  # tied weights

            # biases for autoencoder,  not linked
            self.a_biases1 = tf.Variable(tf.zeros(self.an_hidden1), name="a_biases1")
            self.a_biases2 = tf.Variable(tf.zeros(self.an_hidden2), name="a_biases2")
            self.a_biasescore = tf.Variable(tf.zeros(self.an_hiddencore), name="a_biasescore")
            self.a_biasescoreout = tf.Variable(tf.zeros(self.an_hiddencoreout), name="a_biasescoreout")
            self.a_biases3 = tf.Variable(tf.zeros(self.an_hidden3), name="a_biases3")
            self.a_biases4 = tf.Variable(tf.zeros(self.an_outputs), name="a_biases4")
        
        
        with tf.name_scope("pretraining_Autoencoder"):
            
            
            self.X_flat = tf.contrib.layers.flatten(self.X)
            
            
            self.a_X_drop = tf.layers.dropout(inputs=self.X_flat,rate=self.dropout,training=self.a_training)
            
            
            
            self.a_hidden1 = self.activation(tf.matmul(self.a_X_drop, self.a_weights1) + self.a_biases1)
            self.a_hidden2 = self.activation(tf.matmul(self.a_hidden1, self.a_weights2) + self.a_biases2)
    
            self.a_hiddencore = self.activation(tf.matmul(self.a_hidden2, self.a_weightscore) + self.a_biasescore)
    
            #a_hiddencoreout = activation(tf.matmul(a_stop, a_weightscoreout) + a_biasescoreout)
            self.a_hiddencoreout = self.activation(tf.matmul(self.a_hiddencore, self.a_weightscoreout) + self.a_biasescoreout)
    
            self.a_hidden3 = self.activation(tf.matmul(self.a_hiddencoreout, self.a_weights3) + self.a_biases3)
            self.a_outputs = tf.matmul(self.a_hidden3, self.a_weights4) + self.a_biases4

            self.a_outputs_reshaped = tf.reshape(self.a_outputs, shape=[-1,99,13])
    
            self.a_reconstruction_loss = tf.reduce_mean(tf.square(self.a_outputs - self.X_flat))
            self.a_reg_loss = self.regularizer(self.a_weights1) + self.regularizer(self.a_weights2)
            self.a_loss = self.a_reconstruction_loss + self.a_reg_loss

            self.a_learning_rate = tf.train.natural_exp_decay(self.a_learning_rate_initial,self.a_global_step,self.a_decay_steps,self.a_decay_rate)
            self.a_optimizer = tf.train.AdamOptimizer(self.a_learning_rate)
            self.a_training_op = self.a_optimizer.minimize(self.a_loss, global_step=self.a_global_step)
        
        
        
        self.cells = [
        
        tf.contrib.rnn.GRUCell(num_units=self.n_neurons,activation=tf.nn.tanh), #relu outperforms elu but tanh matches the activation range
        
         
         tf.contrib.rnn.DropoutWrapper(
         tf.contrib.rnn.GRUCell(num_units=(self.n_neurons-10),activation=tf.nn.tanh,  # relus get > 88%
                               kernel_initializer = tf.contrib.layers.xavier_initializer())
          , input_keep_prob = self.input_keep_prob ),
    
        tf.contrib.rnn.DropoutWrapper(
         tf.contrib.rnn.GRUCell(num_units=(self.n_neurons-30),activation= tf.nn.relu, #tf.nn.tanh,  # relus get > 88%
                               kernel_initializer = tf.contrib.layers.xavier_initializer())
          , input_keep_prob =self.input_keep_prob),
    

        tf.contrib.rnn.OutputProjectionWrapper(
            
            tf.contrib.rnn.DropoutWrapper(
         tf.contrib.rnn.GRUCell(num_units=(self.n_neurons-10),activation=tf.nn.relu, #tf.nn.tanh,  # relus get > 88%
                               kernel_initializer = tf.contrib.layers.xavier_initializer())
          , input_keep_prob = self.input_keep_prob),
         
            output_size = self.n_outputs
        )

    
        ]
        
        with tf.name_scope("rnn"):
    
            # 2 layer GRUs work to 90% singleaccurary with some Augmentation on all training data,
            # needs a few more layers to generalize the unwanted labels to unknown
            self.multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.cells, state_is_tuple=False)
    
            # both outputs ( states over times) and states ( final neuron states ) contain information, so using both of those. 
            self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_layer_cell, self.X, dtype=tf.float32)
    
            self.outputs_flat = tf.reshape(self.outputs, [-1, ( self.n_inputs*12)])
        
        
        with tf.name_scope("Conv"):
    

            self.conv_global_step = tf.Variable(0, trainable=False)
            self.a_output_stop_fromconv = tf.stop_gradient(self.a_outputs_reshaped)
            self.input_2d =  tf.expand_dims(self.a_output_stop_fromconv,3)
            self.normresponse_input = tf.nn.local_response_normalization(self.input_2d)
    
            # save memory space by reducing resolution
            self.low_res_input = tf.nn.max_pool(self.input_2d,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding="SAME")

            self.conv_layer = tf.layers.conv2d(self.low_res_input ,
                                          filters=32,
                                          kernel_size = [2,2],
                                          padding='SAME', 
                                  activation=tf.nn.tanh )
 
            self.normresponse_conv_1 = tf.nn.local_response_normalization(self.conv_layer)

            self.max_pool_in = tf.nn.max_pool(self.normresponse_conv_1, ksize=[1,2, 2,1], strides=[1,2, 2,1], padding="SAME",)
    
            self.normresponse = tf.nn.local_response_normalization(self.max_pool_in)
    
            self.conv_layer2 = tf.layers.conv2d(self.normresponse ,filters=32,kernel_size = [3,2],dilation_rate=2, padding='SAME')
            self.conv_layer_elu2 = tf.nn.elu(self.conv_layer2)
    
            self.normresponse_2 = tf.nn.local_response_normalization(self.conv_layer_elu2)
    
            self.conv_layer3 = tf.layers.conv2d(self.normresponse_2 ,filters=32,kernel_size = [3,2],dilation_rate=2, padding='SAME')
            self.conv_layer_elu3 = tf.nn.elu(self.conv_layer3) 
    
            self.normresponse_3 = tf.nn.local_response_normalization(self.conv_layer_elu3)
    
    
            self.pool_flat_in = tf.contrib.layers.flatten(self.normresponse_3)
    
            self.act_bn_norm = tf.layers.batch_normalization(self.pool_flat_in,training=self.train_convNet, momentum = self.momentum)
                             
            self.act_bn_norm_act = tf.nn.elu(self.act_bn_norm)
    
            self.act_elu = tf.layers.dense(self.act_bn_norm_act, units=self.n_neurons, activation=tf.nn.elu)
    
            self.act_drop = tf.layers.dropout(self.act_elu, training = self.train_convNet, rate=self.dropout )
    
            self.act_elu_stage2 = tf.layers.dense(self.act_drop, units=self.n_neurons - 15, activation=tf.nn.elu)
    
            self.act_drop2 = tf.layers.dropout(self.act_elu_stage2, training = self.train_convNet, rate=self.dropout )
    
    
            # outgoing layer, stops the gradient 
            self.conv_stop = tf.stop_gradient(self.act_drop2) # don't propagate the gradient back any further
            self.conv_outgoing = tf.layers.dense(self.conv_stop, units= self.n_neurons - 15 )
    
            # training and evaluation for the Convnet alone
    
            self.conv_logits = tf.layers.dense(self.act_drop2, self.n_outputs, name="conv_outputs_logits") 
        
        
        with tf.name_scope("pretraining_Convolutional"):
            self.conv_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.conv_logits)
            self.conv_loss = tf.reduce_mean(self.conv_xentropy)
            self.conv_optimizer = tf.train.AdamOptimizer(learning_rate=self.conv_learning_rate)# Adamoptimizer implementing the nesterov step 
            self.conv_training_op = self.conv_optimizer.minimize(self.conv_loss)
            self.conv_correct = tf.nn.in_top_k(self.conv_logits, self.y, 1)
            self.conv_accuracy = tf.reduce_mean(tf.cast(self.conv_correct, tf.float32))
    
        
        with tf.name_scope("DenseFromEncoder"):
            self.encoder_stop = tf.stop_gradient(self.a_hiddencore)
    
            self.input_encoder = self.encoder_stop # using the central layer of the encoder
    
            self.first_encoder = tf.layers.dense(self.input_encoder, units=self.n_neurons, activation=tf.nn.elu)

        with tf.name_scope("parted_layer_output"):
    
            # outputs are a tensor of  batch,timepoints(99),cffs(13)
            # feeding this into a 3d convnet need to add channels1  tf.expand_dims(t, 1)
            self.output_2d = tf.expand_dims(self.outputs,3)
            self.low_res_RNN_out = tf.nn.avg_pool(self.output_2d,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding="SAME")
            # reduce resolution
    
    
            self.conv2d = tf.layers.conv2d(self.low_res_RNN_out,filters=20,kernel_size = [3,3],strides=[1,1], padding='SAME')
            self.convoutput_2d_act = tf.nn.elu(self.conv2d)
    
            self.normresponse_out1 = tf.nn.local_response_normalization(self.convoutput_2d_act)
    
            self.conv2d_2 = tf.layers.conv2d(self.normresponse_out1,filters=20,kernel_size = [3,3],strides=[1,1], padding='SAME')
            self.convoutput_2d_act_2 = tf.nn.elu(self.conv2d_2)
    
    
            self.normresponse_out1 = tf.nn.local_response_normalization(self.convoutput_2d_act_2)
            # 99 x 12   => 33 x6
            # 99 x 12  [3,6] => 33*2
            self.max_pool = tf.nn.max_pool(self.normresponse_out1, ksize=[1,3, 3,1], strides=[1,3, 3,1], padding="SAME")

            self.pool_flat = tf.contrib.layers.flatten(self.max_pool)
    
            self.dense_out = tf.layers.dense(self.pool_flat, (self.n_neurons), #activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name="supplementationDense_fullRNNoutput") # add a dense layer
                         
        with tf.name_scope("parted_layer_states"):   
    
            self.dense_state = tf.layers.dense(self.states, (self.n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name="supplementationDense_covering_states") # add a dense layer
    
    
        with tf.name_scope("combine_states"):
    
    
            self.combinedstates = tf.concat([self.dense_state, self.dense_out,self.conv_outgoing], 1)

        
        with tf.name_scope("supplement_layer"):
   
   
            self.dense2 = tf.layers.dense(self.combinedstates, (self.n_neurons * 2 ), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.scale),
                           name="supplementationDense2") # add a dense layer
    
    #batch normalization  
    # doesn't improve output applied before skip connections
    
    
            self.bn_norm_sup2 = tf.layers.batch_normalization(self.dense2,training=self.training, momentum = self.momentum)
                             
            self.bn_norm_act2 = tf.nn.elu(self.bn_norm_sup2) # activate the batch norm
    
    
     # combine with states here 
        
            self.drop2_ext = tf.concat([self.bn_norm_act2, self.states], 1)
    
    # addition
 
    
            self.bn_norm_sup_ext = tf.layers.batch_normalization(self.drop2_ext,training=self.training, momentum = self.momentum)
                             
            self.bn_norm_act_ext = tf.nn.elu(self.bn_norm_sup_ext)
    
    
    
    
            self.dense3 = tf.layers.dense(self.bn_norm_act_ext, (self.n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.scale),
                           name="supplementationDense3") # add a dense layer
    
    
    # add autoencoder data here
    
            self.dense_encoded = tf.concat([self.dense3, self.first_encoder], 1)
    
    
            self.bn_norm_enc_ext = tf.layers.batch_normalization(self.dense_encoded,training=self.training, momentum = self.momentum)
                             
            self.bn_norm_enc_ext = tf.nn.elu(self.bn_norm_enc_ext)
    
            self.dense3_encode = tf.layers.dense(self.bn_norm_enc_ext, (self.n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.scale),
                           name="supplementationDense3enc") # add a dense layer
    
            self.drop3 = tf.layers.dropout(self.dense3_encode, training = self.training, rate=(1-self.input_keep_prob))
    
            self.dense3_b = tf.layers.dense(self.drop3, (self.n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.scale),
                           name="supplementationDense3_b") # add a dense layer
    
            self.bn_norm_sup3 = tf.layers.batch_normalization(self.dense3_b,training=self.training, momentum = self.momentum)
                             
            self.bn_norm_act3 = tf.nn.elu(self.bn_norm_sup3) # activate the batch norm
    
    
    
     # combine with conv_part here

            self.drop3_ext = tf.concat([self.dense3, self.conv_outgoing], 1)
    
    #addition
    
            self.dense4 = tf.layers.dense(self.drop3_ext, (self.n_neurons - 10), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.scale),
                           name="supplementationDense4") # add a dense layer
    
            self.bn_norm_sup4 = tf.layers.batch_normalization(self.dense4,training=self.training, momentum = self.momentum)                  
            self.bn_norm_act4 = tf.nn.elu(self.bn_norm_sup4) # activate the batch norm
    
    
            self.drop4 = tf.layers.dropout(self.bn_norm_act4, training = self.training, rate=(1-self.input_keep_prob) )
    
    # combine with output here
    
            self.dense4_ext = tf.concat([self.drop4, self.states], 1)
    
  
            self.dense5 = tf.layers.dense(self.dense4_ext, (self.n_neurons - 55), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.scale),
                           name="supplementationDense5") # add a dense layer
    
    
            self.drop5 = tf.layers.dropout(self.dense5, training = self.training, rate=self.dropout )
    
            self.dense6 = tf.layers.dense(self.drop5, (self.n_neurons - 55), activation=tf.nn.elu,  
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.scale),
                           name="supplementationDense6") # add a dense layer
        
        with tf.name_scope("logits"):

            self.logits_before_bn = tf.layers.dense(self.dense6, self.n_outputs, name="outputs_logits") #logits = tf.layers.dense(states, n_outputs)
            self.logits = tf.layers.batch_normalization(self.logits_before_bn,training=self.training, momentum = self.momentum)
    
    

        #  for training of the RNN
        with tf.name_scope("loss"):
    
    
    
            # sparse softmax, so I don't have to get the ohot
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
    
            # regularization losses
            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
            # ignore regularization  it's not helping
            self.loss = tf.reduce_mean(self.xentropy)
            #
            #  activate regularization here
            self.loss_aftereg = tf.add_n([self.loss]+ self.reg_loss)
    
    
        with tf.name_scope("train"):
    
    
            self.global_step = tf.Variable(0, trainable=False)

            self.learning_rate = tf.train.exponential_decay(self.initial_learning,self.global_step,self.decay_steps,self.decay_rate)
            #learning_rate = tf.train.natural_exp_decay(initial_learning,global_step,decay_steps,decay_rate)
    
            #learning_rate = 0.00074
            self.optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate) # Adamoptimizer implementing the nesterov step 
            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate) # recommended for RNN,  not optimal performance
            #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
            #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
            training_op = self.optimizer.minimize(self.loss_aftereg, global_step=self.global_step)

        with tf.name_scope("eval"):
            self.correct = tf.nn.in_top_k(self.logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
    
        with tf.name_scope("logging"):
    
            self.accuracy_sum = tf.summary.scalar('accuracy',self.accuracy)
            self.loss_sum = tf.summary.scalar('loss',self.loss)
    
            self.summaries = tf.summary.merge_all()
    
            # writ to tensorboard
            self.train_writer = tf.summary.FileWriter(self.logdir + '/train', tf.get_default_graph())
            self.test_writer = tf.summary.FileWriter(self.logdir + '/test')
    
    
        with tf.name_scope("predict"):
            self.prediction = tf.argmax(self.logits,1)
    
        
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    
    def run_autoencoder(self, prefetched_test, X_mess, batch_size = 320, n_epochs = 25, verbose = False,):
        with tf.Session() as sess:
            self.init.run()
            for epoch in range(n_epochs):
                for iteration in range( prefetched_test.shape[0]// (batch_size) ):
                    X_batch, y_batch = batch_from_prefetch_samples(prefetched_test,batch_size)
                    #X_batch = X_batch.tolist()


                    sess.run(a_training_op, feed_dict={self.Aenc_training: True, self.X: X_batch})
        
        
                loss_train = self.a_reconstruction_loss.eval(feed_dict={self.X: X_batch})
                loss_val = self.a_reconstruction_loss.eval(feed_dict={self.X: X_mess})
                
                if verbose:
                    print("\r{}".format(epoch), "Train MSE:", loss_train, " val MSE:", loss_val)    
        
            saver.save(sess, "FirstStep_Autoencoderonly_%s.ckpt" %epoch)


    def run_convnet(self, autoencoder_savepath, prefetched_data, X_val, y_val , batch_size = 300 , n_epochs= 420, verbose = True):
        print("convnet")
        with tf.Session() as sess:
            #init.run()
            saver.restore(sess,autoencoder_savepath) 

            for epoch in range(n_epochs):
                for iteration in range( index_df.shape[0] // (batch_size) ):
                    X_batch, y_batch = batch_from_prefetch_samples(prefetched_data,batch_size)
                    
            
                    sess.run([self.conv_training_op], feed_dict={self.train_convNet : True, self.X: X_batch, self.y: y_batch, })
            
            
                acc_train = self.accuracy.eval(feed_dict={self.X: X_test, self.y: y_test})
                acc_test = self.accuracy.eval(feed_dict={self.X: X_test, self.y: y_test})
        
                if acc_test >0.6:
                    save_path = saver.save(sess, "Conv_checkpoint/Intermed_conv_auto_%s.ckpt" %acc_test)
            
                if verbose:
                    print(epoch, " Conv Train accuracy:", acc_train, " test" , acc_test)
         
            save_path = saver.save(sess, "Second_Stage_conv_auto_%s.ckpt" %(n_epochs) )
        
    def run(self, convnet_savepath,prefetched_data, X_val, y_val, verbose = False, n_epochs = 420, batch_size = 300):
        with tf.Session() as sess:
            init.run()
            count = 0
    
    # load the autoencoder to pre-process the data
            saver.restore(sess,convnet_savepath) 
   
    
            for epoch in range(n_epochs):

                for iteration in range( index_df.shape[0] // (batch_size * 8) ):
                    X_batch, y_batch = batch_from_prefetch_samples(prefetched_data,batch_size)
                    X_batch_aug = augment_batch(X_batch= X_batch, noise_max=0.01, shift_max=8)
                    sess.run([self.training_op,self.extra_update_ops], feed_dict={self.training: True, self.X: X_batch_aug, self.y: y_batch})
            
                    if iteration % 200 == 0:
                        step = epoch*iteration
                        summary, _ = sess.run([self.summaries,self.extra_update_ops], feed_dict={self.X: X_test, self.y: y_test} )
                        self.train_writer.add_summary(summary, step )

                        summary, _ = sess.run([self.summaries,self.extra_update_ops], feed_dict={self.X: X_val, self.y: y_val})
                        self.test_writer.add_summary(summary,step)
                        
                        acc_train = self.accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                        acc_val = self.accuracy.eval(feed_dict={X: X_val, y: y_val})
        
                step = epoch*iteration
                summary, _ = sess.run([self.summaries,self.extra_update_ops], feed_dict={self.X: X_test, self.y: y_test} )
                self.train_writer.add_summary(summary, step )     
                summary, _ = sess.run([self.summaries,self.extra_update_ops], feed_dict={self.X: X_val, self.y: y_val})    
                self.test_writer.add_summary(summary,step)
        
        
      
                #if count % 5 == 0:
                #    save_path = saver.save(sess, "RNN_4_concat_conv2d_ampdata_1001_savedbycount_%s_%s.ckpt" %(epoch,acc_mess) )
                #count +=1
            
                acc_val = self.accuracy.eval(feed_dict={self.X: X_val, self.y: y_val}) 
                
                val_pred = self.prediction.eval(feed_dict={self.X: X_val})
                multi_acc = multiclass_accuracy(y_val,val_pred)
        
                if multi_test > 0.98:
                    save_path = saver.save(sess, "RNN__%s.ckpt" %(multi_acc) )
                    print("written savepoint %s " %(multi_acc))
        
                print(epoch, "Train accuracy:", round(acc_train,2), " - ", round(acc_test,2),"mediocre :", "ma test: ", round(multi_test,2))
            
            
            save_path = saver.save(sess, "Final_training_auto_conv_RNN.ckpt")

        
        
    def predict(self,  checkpoint = "Final_training_auto_conv_RNN.ckpt"):
        print("predicting")
        saver.restore(sess, "Final_training_auto_conv_RNN.ckpt")

        reslist = []
    
        for group in chunker(data,1000):
            results = prediction.eval(feed_dict={X:group})
            for element in results.tolist():
                reslist.append(element)
        self.reslist = reslist


# In[82]:




# In[83]:




# In[ ]:



