# 	Mentor:Dr. Mustaq Ahmed
#	Akshay Arora,Anand Mukut Tirkey, Harivamshi Valamkonda
#!/usr/bin/python
# -*- coding: utf-8 -*-
# 3 layer neural network 




import glob
import os
import librosa  #removed .display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pylab
import csv
from matplotlib.pyplot import specgram
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import matplotlib


#Function to read feature vectors and corresponding labels from 
#csv files  
def read_vector():

#    features, labels = np.empty((0,193)), np.empty(0)

# reading files contianing features 

    f = open('heartftr.csv', 'r')
    f1 = open('heartlbl.csv', 'r')
    features = genfromtxt(f, delimiter=',')
    labels = genfromtxt(f1, delimiter=',')
    #print features, labels
    return (np.array(features), np.array(labels, dtype=np.int))


####Feature engineering
#Function uses librosa library and extract various features
#such as mfcc,chroma,melspectogram,spectral_contrast etc.
def extract_feature(file_name):
	# reading the audio file
    (X, sample_rate) = librosa.load(file_name)

    #short time furiour transform (STFT)
    stft = np.abs(librosa.stft(X))

    #mel frequency cepstral coefficient
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                    n_mfcc=40).T, axis=0)

    #compute a chromagram from a waveform or power spectrum
    chroma = np.mean(librosa.feature.chroma_stft(S=stft,
                     sr=sample_rate).T, axis=0)

    #compute a mel-scaled spectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,
                  axis=0)

    #computing spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft,
                       sr=sample_rate).T, axis=0)

    #computes the tonal centroid feature (tonnetz)
    tonnetz = \
        np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                sr=sample_rate).T, axis=0)
    return (mfccs, chroma, mel, contrast, tonnetz)



#Function to parse audio files and access database to extract labels
#call extract features for each audio file

def parse_audio_files(parent_dir, sub_dirs, file_ext='.wav'):
    (features, labels) = (np.empty((0, 193)), np.empty(0))
    f = open('heartftr.csv', 'a+')
    f1 = open('heartlbl.csv', 'a+')
    for (label, sub_dir) in enumerate(sub_dirs):
        cnt = 0
        filename_label = []
        with open(parent_dir + sub_dir + 'REFERENCE.csv', 'r') as \
            csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                filename_label.append(row)
        for fn in filename_label:
            try:
                print fn
                (mfccs, chroma, mel, contrast, tonnetz) = \
                    extract_feature(parent_dir + sub_dir + fn[0]
                                    + file_ext)
            except Exception, e:
                print 'Error encountered while parsing file: ', fn
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast,
                    tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, (int(fn[1])+1)/2)
    np.savetxt(f, features, delimiter=',')
    np.savetxt(f1, np.array(labels, dtype=np.int), delimiter=',')
    print features, ' # ', labels
    return (np.array(features), np.array(labels, dtype=np.int))



#convert labels from integer to binary representation

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def randomize_data (feat,labe):
	for old_index in range(0,len(feat)):
		new_index = np.random.randint(old_index+1)
		feat[old_index],feat[new_index] = feat[new_index],feat[old_index]
		labe[old_index],labe[new_index] = labe[new_index],labe[old_index]
	return feat,labe


parent_dir = 'data/'
tr_sub_dirs = ['training-a/', 'training-b/', 'training-c/',
               'training-d/', 'training-e/']
ts_sub_dirs = ['training-f/']

#(tr_features, tr_labels) = parse_audio_files(parent_dir, tr_sub_dirs)
#(ts_features, ts_labels) = parse_audio_files(parent_dir, ts_sub_dirs)

#reading saved feature vector
dataf,datal=read_vector()

dataf,datal = randomize_data(dataf,datal)


#dataf=dataf[0:1000]
#datal=datal[0:1000]
#visulaization of data
#vis_data = bh_sne(dataf)
#vis_x = vis_data[:, 0]
#vis_y = vis_data[:, 1]

# colors = ['red','green']

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(dataf)


# plt.scatter(tsne_results[:,0], tsne_results[:,1], c=datal, cmap=matplotlib.colors.ListedColormap(colors))
# # plt.colorbar(ticks=range(10))
# # plt.clim(-0.5, 9.5)
# plt.show()






#allocating training and testing data
tr_features,tr_labels=dataf[1001:],datal[1001:]
#tr_features=np.vstack([tr_features,dataf[2501:]])
#tr_labels=np.hstack([tr_labels,datal[2501:]])

ts_features, ts_labels=dataf[0:1000],datal[0:1000]

# print tr_features,tr_labels
# print ts_features, ts_labels
# print "#",tr_features,tr_labels
tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)

#print tr_labels, ts_labels


#parameters initialization
training_epochs = 5000
n_dim = tr_features.shape[1]
n_classes = 2
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.0001


##seting up input and output dimension of numpy array
## 3 layer neural network
## seting up weights and biases randomly

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0,
                  stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0,
                  stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1) 

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,
                  n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0,
                  stddev=sd))
h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2) 

W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes],
                mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

# initialize tensorflow variables
init = tf.global_variables_initializer()


#cost function to minimize
cost_function = -tf.reduce_sum(Y * tf.log(y_)+0.001)
optimizer = \
    tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)
(y_true, y_pred) = (None, None)


#### Training model
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        (_, cost) = sess.run([optimizer, cost_function],
                             feed_dict={X: tr_features, Y: tr_labels})
        #print _, cost, epoch
        print epoch
        cost_history = np.append(cost_history, cost)

    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels, 1))
    print y_pred, y_true
    print ('Test accuracy: ', round(sess.run(accuracy,
           feed_dict={X: ts_features, Y: ts_labels}), 3))

fig = plt.figure(figsize=(10, 8))
plt.plot(cost_history)
plt.axis([0, training_epochs, 0, np.max(cost_history)])
plt.show()

# (p, r, f, s) = precision_recall_fscore_support(y_true, y_pred,
#         average='micro')
# print 'F-Score:', round(f, 3)


			
