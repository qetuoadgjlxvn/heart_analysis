import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cPickle
import csv
import copy

log_specgrams = []
log_labels=[]
def read_vector():
#    features, labels = np.empty((0,193)), np.empty(0)
    f=open("urbanftr2.csv","r")
    f1=open("urbanlbl2.csv","r")
    log_specgrams = genfromtxt(f, delimiter=',')
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    labels = genfromtxt(f1, delimiter=',')
   # print features,labels
    return np.array(features), np.array(labels, dtype = np.int)


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract2(fn,window_size,lbl,bands=60):
    
    sound_clip,s = librosa.load(fn)

    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.logamplitude(melspec)
            #print logspec
            logspec = logspec.T.flatten()[:, np.newaxis].T
            #print "#################",logspec
            log_specgrams.append(logspec)
            log_labels.append(int(lbl))
            #cPickle.dump( log_specgrams, open( "filename.pkl", "wb" ) )
    #return log_specgrams
    
def extract_features(parent_dir,sub_dirs,flag,file_ext=".wav",bands = 60, frames = 41):
    print('hello')
    window_size = 512 * (frames - 1)
    final = []
    #labels = []
    '''f=open("urbanftr2.csv","a+")
    f1=open("urbanlbl2.csv","a+")'''
 
    for l, sub_dir in enumerate(sub_dirs):
        print("l is",l)
        filename_label = []
        with open(parent_dir + sub_dir + 'REFERENCE.csv', 'r') as \
            csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                filename_label.append(row)
        for fn in filename_label:
            try:
                print fn
                extract2(parent_dir + sub_dir + fn[0]
                                    + file_ext,window_size,(int(fn[1])+1)/2)
                #log_specgrams.append(temp)
                #cPickle.dump( log_specgrams, open( "filename.pkl", "wb" ) )
            except Exception, e:
                print 'Error encountered while parsing file: ', fn
                continue
            '''ext_features = np.hstack([mfccs, chroma, mel, contrast,
                    tonnetz])
            features = np.vstack([features, ext_features])'''
            #labels = np.append(labels, (int(fn[1])+1)/2)
            #labels.append((int(fn[1])+1)/2)
        print fn
            #print log_specgrams
    #np.savetxt(f,log_specgrams,delimiter=",")
    final = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((final, np.zeros(np.shape(final))), axis = 3)

    #np.savetxt(f1,np.array(labels, dtype = np.int),delimiter=",")

    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    '''if flag==0:
        #cPickle.dump( features, open( "tr_features.pkl", "wb" ) )
        cPickle.dump( labels, open( "tr_labels.pkl", "wb" ) )
    else:
        #print("nothing")
        #cPickle.dump( features, open( "ts_features.pkl", "wb" ) )
        cPickle.dump( labels, open( "ts_labels.pkl", "wb" ) )'''

    return np.array(features), np.array(log_labels,dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    print type(labels),labels
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
print "pickle "  



#tr_features = cPickle.load( open( "tr_features.pkl", "rb" ) )
#tr_labels = cPickle.load( open( "tr_labels.pkl", "rb" ) )
#ts_features = cPickle.load( open( "ts_features.pkl", "rb" ) )
#ts_labels = cPickle.load( open( "ts_labels.pkl", "rb" ) )

#parent_dir = 'UrbanSound8K/audio'
parent_dir = 'data/'
tr_sub_dirs = ['training-a/', 'training-b/', 'training-c/',
               'training-d/', 'training-e/']
#tr_sub_dirs = ['training-d/']
ts_sub_dirs = ['training-f/']
tr_features_temp,tr_labels_temp = extract_features(parent_dir,tr_sub_dirs,0)

tr_features=copy.deepcopy(tr_features_temp)
tr_labels=copy.deepcopy(tr_labels_temp)
#ts_sub_dirs= ['fold3']
tr_labels = one_hot_encode(tr_labels)

log_specgrams[:] = []
log_labels[:]=[]
ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs,1)
#ts_features,ts_labels= tr_features,tr_labels
ts_labels = one_hot_encode(ts_labels)
#tr_labels=ts_labels




print np.shape(ts_features),np.shape(ts_labels),"$"
print np.shape(tr_features),np.shape(tr_labels),"$"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], 
                          strides=[1, stride_size, stride_size, 1], padding='SAME')


frames = 41
bands = 60

feature_size = 2460 #60x41
num_labels = 2
num_channels = 2

batch_size = 50
kernel_size = 30
depth = 20
num_hidden = 200

learning_rate = 0.01
total_iterations = 2000



X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

cov = apply_convolution(X,kernel_size,num_channels,depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])
print np.shape(cov_flat),shape
f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)


loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
with tf.Session() as session:
    tf.global_variables_initializer().run()

    for itr in range(total_iterations):    
        offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
        batch_x = tr_features[offset:(offset + batch_size), :, :, :]
        batch_y = tr_labels[offset:(offset + batch_size), :]
        
        print np.shape(batch_x),np.shape(batch_y),"#",itr
        _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
        cost_history = np.append(cost_history,c)
        
    
    print('Test accuracy: ',round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}) , 3))
    fig = plt.figure(figsize=(10,8))
    plt.plot(cost_history)
    plt.axis([0,total_iterations,0,np.max(cost_history)])
    plt.show()



