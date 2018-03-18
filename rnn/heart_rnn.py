import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import csv
import cPickle


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext=".wav",bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        filename_label = []
        

        with open(parent_dir + sub_dir + 'REFERENCE.csv', 'r') as \
            csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                filename_label.append(row)
        for fn in filename_label:
            fnm=parent_dir + sub_dir + fn[0]+ file_ext
            label=(int(fn[1])+1)/2
            print fnm,label
            sound_clip,s = librosa.load(fnm)
            #log_specgrams.append(temp)
            #cPickle.dump( log_specgrams, open( "filename.pkl", "wb" ) )
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)         
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)
    cPickle.dump( features, open( "rnn_features.pkl", "wb" ) )
    cPickle.dump( labels, open( "rnn_labels.pkl", "wb" ) )

    return np.array(features), np.array(labels,dtype = np.int)




def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
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
               'training-d/', 'training-e/','training-f/']
#tr_sub_dirs = ['training-d/']               
#ts_sub_dirs = ['training-f/']


tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
#ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs,1)



read_features = cPickle.load( open( "rnn_features.pkl", "rb" ) )
read_label = cPickle.load( open( "rnn_labels.pkl", "rb" ) )

read_features,read_label=randomize_data(read_features,read_label)


tr_features,tr_labels=read_features[1001:],read_label[1001:]
ts_features,ts_labels=read_features[:1000],read_label[:1000]


tr_labels = one_hot_encode(tr_labels)


ts_labels = one_hot_encode(ts_labels)










tf.reset_default_graph()

learning_rate = 0.01
training_iters = 1000
batch_size = 50
display_step = 200
epoch = 100

# Network Parameters
n_input = 20 
n_steps = 41
n_hidden = 300
n_classes = 2

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes])) 


# def RNN(x, weight, bias):
#     cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
#     #cell = tf.nn.rnn_cell.MultiRNNCell([ltsm] * 2, state_is_tuple=True)
#     layer1, _istate = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
#     layer1 = tf.reshape(layer1, shape=[batch_size, rows*n_hidden])
#     layer2 = tf.nn.dropout(layer1, proba)
#     return tf.matmul(layer2, layer1_weights) + layer1_biases


def RNN(x, weight, bias):
    cell = rnn_cell.BasicLSTMCell(n_hidden,state_is_tuple = True)
        
    cell=rnn_cell.DropoutWrapper(cell=cell,output_keep_prob=0.75)
    #cell = rnn_cell.MultiRNNCell([cell] * 2)

   # print np.shape(x),cell
    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)


prediction = RNN(x, weight, bias)

# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init)
    
    for itr in range(training_iters):    
        offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
        batch_x = tr_features[offset:(offset + batch_size), :, :]
        batch_y = tr_labels[offset:(offset + batch_size), :]
        _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
            
        if epoch % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
    
    print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}) , 3))
