"""
Description: 
    This script trains the Transformer on spectrogram data
    We don't use slicing or positional encoding here
    This achieves around 39% overall accuracy accross SNRs
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
import h5py
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform
from scipy import signal


# Extracts data from RadioML dataset
maxlen = 128 # change subnyq sampling rate HERE.
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


# Converts I/Q data to complex form
# We need to do this first before converting to spectrograms
def to_complex(xtrain, xtest):
    X_train_cmplx = xtrain[:,0,:] + 1j* xtrain[:,1,:]
    X_test_cmplx = xtest[:,0,:] + 1j* xtest[:,1,:]
    return (X_train_cmplx, X_test_cmplx)
xtrain1, xtest1 = to_complex(xtrain1, xtest1)
print(xtrain1.shape)

# Creates new np arrays to hold new complex data
X_train = np.zeros((600000, 128), dtype='complex_')
X_test = np.zeros((600000, 128), dtype='complex_')
for i in range(len(xtrain1)):
    X_train[i] = xtrain1[i].flatten()
for i in range(len(xtest1)):
    X_test[i] = xtest1[i].flatten()
Y_train = ytrain1
Y_test = ytest1


# Converts I/Q data to spectrogram form
class ChannelIndSpectrogram():
    def __init__(self,):
        pass
    
    def _normalization(self,data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)
        
        for i in range(data.shape[0]):
        
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i]/rms
        
        return s_norm        

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
    
        return x_cropped

    # win_len and overlap parameters are suited for our data, might be able to change them
    def _gen_single_channel_ind_spectrogram(self, sig, win_len=4, overlap=2):
        '''
        _gen_single_channel_ind_spectrogram converts the IQ samples to a channel
        independent spectrogram according to set window and overlap length.
        
        INPUT:
            SIG is the complex IQ samples.
            
            WIN_LEN is the window length used in STFT.
            
            OVERLAP is the overlap length used in STFT.
            
        RETURN:
            
            CHAN_IND_SPEC_AMP is the genereated channel independent spectrogram.
        '''
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig)
        
        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)
        
        # Generate channel independent spectrogram.
        chan_ind_spec = spec[:,1:]/spec[:,:-1]    
        
        # Take the logarithm of the magnitude.      
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec)**2)
                  
        return chan_ind_spec_amp
    


    def channel_ind_spectrogram(self, data):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent 
        spectrograms.
        
        INPUT:
            DATA is the IQ samples.
            
        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''
        
        # Normalize the IQ samples.
        data = self._normalization(data)
        
        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = 2
        num_column = int(np.floor((data.shape[1]-4)/2 + 1) - 1) # 4, 2, and 52 are proper for our data
        data_channel_ind_spec = np.zeros([num_sample, 52, 2, 1])
        #print(data_channel_ind_spec.shape)
        
        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):          
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i,:,:,0] = chan_ind_spec_amp
            
        return data_channel_ind_spec

# Create class and X sets
ChannelIndSpectrogramObj = ChannelIndSpectrogram()
# The input 'data' is the loaded IQ samples in the last example.
ch_ind_spec_train = ChannelIndSpectrogramObj.channel_ind_spectrogram(X_train)
ch_ind_spec_test = ChannelIndSpectrogramObj.channel_ind_spectrogram(X_test)
print(ch_ind_spec_train.shape)

# Reshape to proper shape for training
x_spec_train = ch_ind_spec_train.reshape(600000, 2, 52)
x_spec_test = ch_ind_spec_test.reshape(600000, 2, 52)


# Train transformer like usual
embed_dim = 52  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 52 # Hidden layer size in feed forward network inside transformer
in_shp = x_spec_train.shape[1:]

inputs = Input(shape=(in_shp))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(inputs)
x = Flatten()(x)
x = Dropout(0.1)(x)
x = Dense(52, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(10, activation="softmax")(x)

models = Model(inputs=inputs, outputs=outputs)

opt = Adam(learning_rate=0.001)
models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
num_epochs = 500
early = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto')

history = models.fit(x_spec_train,
                    Y_train,
                    epochs=num_epochs,
                    batch_size=1024,
                    verbose=2,
                    callbacks = [early],
                    validation_split=0.25)