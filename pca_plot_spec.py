"""
Description: 
    This script plots the spectrogram data with PCA
    Useful for seeing how well the data can be clustered

Important notes:
    Make sure you adjust certain variables like the one on line 45 for your purposes
"""

import numpy as np
import pickle 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

    # IMPORTANT
    n_train = n_examples // 1024 # this '1024' value can changed based on how much data you want to plot
                                # the bigger the number, the less data you'll plot

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
# IMPORTANT
# This '1171' is the number of data samples in your xtrain
# Change it based on line 45
X_train = np.zeros((1171, 128), dtype='complex_')

for i in range(len(xtrain1)):
    X_train[i] = xtrain1[i].flatten()
Y_train = ytrain1


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
print(ch_ind_spec_train.shape)

# Reshape data to proper form for plotting
# IMPORTANT
# This '1171' value needs to change based on line 45
x_spec_train = ch_ind_spec_train.reshape(1171, -1)
data = x_spec_train

#Scale the data
scaler = StandardScaler()
scaler.fit(data)
scaled = scaler.transform(data)

#Obtain principal components
pca = PCA().fit(scaled)
pc = pca.transform(scaled)
pc1 = pc[:,0]
pc2 = pc[:,1]

#Plot principal components
plt.figure(figsize=(10,10))

# These parameters should be changed to test certain classes
# Assigns colors to certain classes
# White represents every other data sample
# y[6] means we are choosing the 7th class in the y set
colour = ['b' if y[6]==1 else 'g' if y[8]==1 else 'r' if y[-1]==1 else 'w' for y in Y_train]
plt.scatter(pc1,pc2 ,c=colour,edgecolors='#000000')
plt.ylabel("Glucose",size=20)
plt.xlabel('Age',size=20)
plt.yticks(size=12)
plt.xticks(size=12)
plt.xlabel('PC1')
plt.ylabel('PC2')

# Plots PCA scree plot
var = pca.explained_variance_[0:10] #percentage of variance explained
labels = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
plt.figure(figsize=(15,7))
plt.bar(labels,var,)
plt.xlabel('Pricipal Component')
plt.ylabel('Proportion of Variance Explained')