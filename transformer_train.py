"""
Description: 
    This script trains a Transformer network on the RadioML2016 dataset
    It achieves 75% classification accuracy at SNR=18
"""

import numpy as np
import pickle 
import random 
import matplotlib.pyplot as plt
from transformer_class import *
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, Reshape, GlobalAveragePooling1D
from tensorflow.keras import metrics
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
from numpy import linalg as la
from math import ceil
from tensorflow.keras.callbacks import LearningRateScheduler
import time

maxlen = 128 # change subnyq sampling rate HERE.

# Zero padding
def norm_pad_zeros(X_train,nsamples):
    
    print("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/la.norm(X_train[i,:,0],2)
    return X_train

# Converts I/Q data to amp-phase/polar form
def to_amp_phase(X_train,X_test,nsamples):
    X_train_cmplx = X_train[:,0,:] + 1j* X_train[:,1,:]
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]
    
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:,1,:],X_train[:,0,:])/np.pi
    
    X_train_amp = np.reshape(X_train_amp,(-1,1,nsamples))
    X_train_ang = np.reshape(X_train_ang,(-1,1,nsamples))
   
    X_train = np.concatenate((X_train_amp,X_train_ang), axis=1) 
    X_train = np.transpose(np.array(X_train),(0,2,1))
    
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi
    
    
    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))
    
    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1) 
    X_test = np.transpose(np.array(X_test),(0,2,1))
    return (X_train, X_test)

# Extracts data from RadioML dataset
def gendata(fp, nsamples):
    global snrs, mods, train_idx, test_idx, lbl
    with open(fp, 'rb') as p:
        Xd = pickle.load(p, encoding='latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    print(mods, snrs)
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)
    
    print('Length of lbl', len(lbl))
    print('shape of X', X.shape)

    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = n_examples // 2
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1
    X_train = X[train_idx]
    X_test =  X[test_idx]
    keys = Xd.keys()
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    return (X_train,X_test,Y_train,Y_test)

# Assigns data to proper variables
xtrain1,xtest1,ytrain1,ytest1 = gendata("./data/RML2016.10b.dat",maxlen)
print('using version 10b dataset')

# Extracting SNR/class data
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
train_snr = lambda snr: xtrain1[np.where(np.array(train_SNRs)==snr)]
test_snr = lambda snr: ytrain1[np.where(np.array(train_SNRs)==snr)]
classes = mods

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conversion to amp-phase form ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('length of X before to_amp_phase:', xtrain1.shape)
xtrain1,xtest1 = to_amp_phase(xtrain1,xtest1,maxlen)
print('length of X after to_amp_phase:', xtrain1.shape)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Add padding
xtrain1 = xtrain1[:,:maxlen,:]
xtest1 = xtest1[:,:maxlen,:]
xtrain1 = norm_pad_zeros(xtrain1,maxlen)
xtest1 = norm_pad_zeros(xtest1,maxlen)

# Reshape to form that can be used as Transformer model input
# These lines must remain in order to train properly
X_train = np.reshape(xtrain1, (-1,2,128))
X_test = np.reshape(xtest1, (-1,2,128))

# Reshape Y sets to proper shape
Y_train = np.reshape(ytrain1,(-1,10))
Y_test = np.reshape(ytest1,(-1,10))

# Useful print outs
print("Training data:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*50)

# Hyperparameters exclusively for the Transformer
# Only num_heads variable should be changed
embed_dim = 128  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer
in_shp = X_train.shape[1:]

# Create Transformer model by layers
inputs = Input(shape=(in_shp))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim) # Encoder
x = transformer_block(inputs)
x = Flatten()(x)
x = Dropout(0.1)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(len(classes), activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

# Optional learning rate scheduler to optimize training
# Would use "scheduler" in "callbacks" part of model.fit
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Compile model with hyperparameters
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
num_epochs = 500
early = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto')
lr_scheduler = LearningRateScheduler(scheduler)

# Start training
start = time.time()
history = model.fit(X_train,
                    Y_train,
                    epochs=num_epochs,
                    batch_size=1024,
                    verbose=2,
                    callbacks = [early],
                    validation_split=0.25)
end = time.time()
print(end - start)

# Test the network
acc={}
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    print(test_SNRs[:3])
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy for SNR = " + str(snr) + ": ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
print(acc)