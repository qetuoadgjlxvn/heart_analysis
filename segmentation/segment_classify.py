
import os
import librosa
import numpy as np
import pandas as pd
from librosa import display
import matplotlib.pyplot as plt
from keras.layers import InputLayer, Conv1D, Dense, Flatten, MaxPool1D
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import copy

PATH='/home/user/Desktop/heart_sound_segmentation/heartbeat-sounds/set_a'


def get_files_in_directory(directory):
	for filename in os.listdir(directory):
		print filename

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


def get_files_in_directory(f_test_x,f_test_y,model):
	

	for j in range(0,len(f_test_x)):
		x = model.predict(f_test_x[j])
		i = 0
		cnt=0
		for data in x:
			z=np.max(data)
			print z
			if(z==data[f_test_y[j][i]]):
				print 'correct'
				cnt+=1
			print "prediction ",data,f_test_y[j][i]
			i+=1

		print "result ",cnt," ",i


def read_segment(filename,model):
	temp = pd.read_csv(filename)
# 	data_x=[]
# 	data_y=[]
# 	for j in range(0,(2*temp.shape[0])/3):
# 		for i in range(1, temp.shape[1] - 1):
# 			try: 
# 				print PATH+ temp.iloc[j, 0].split('.')[0] +'.wav'
# 				data, sampling_rate = librosa.load(PATH + temp.iloc[j, 0].split('.')[0] +'.wav', sr=44100 )
# 				temp_data = data[int(temp.iloc[j, i]):int(temp.iloc[j, i+1])]
# 				temp_label = temp.iloc[j, 0].split('__')[0]
# #NOT USED YET
# #				(mfccs, chroma, mel, contrast, tonnetz) = extract_feature(temp_data,sampling_rate)
# 				data_x.append(temp_data)
# 				data_y.append(temp_label)
# 			except:
# 				pass

# 	data_x = pad_sequences(data_x, maxlen=20000, dtype='float', padding='post', truncating='post', value=0.)
# 	data_x = data_x / np.max(data_x)
# 	# data_y.append('normal')
# 	# data_y.append('extrahls')
# 	# data_y.append('artifact')
# 	# data_y.append('murmur')
	

# 	#data_x = data_x[:,:,np.newaxis]
# 	data_y = pd.Series(data_y)
# 	data_y.value_counts()

	

# 	data_y = data_y.map({'/normal':0, '/extrahls':1,'/artifact':2,'/murmur':3}).values
# #	print data_y
# 	print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"	
# #	test_x = copy.deepcopy(data_x[:len(data_x)/3])
# #	test_y = copy.deepcopy(data_y[:len(data_y)/3])
# #	data_x=data_x[len(data_x)/3:]
# #	data_y=data_y[len(data_y)/3:]


# 	encoder = LabelEncoder()
# 	encoder.fit(data_y)
# 	encoded_Y = encoder.transform(data_y)
# 	# convert integers to dummy variables (i.e. one hot encoded)
# 	dummy_y = np_utils.to_categorical(encoded_Y)

# 	print data_x,'##############',data_y,dummy_y
# 	# model = Sequential()

# 	# model.add(InputLayer(input_shape=data_x.shape[1:]))

# 	# model.add(Conv1D(filters=50, kernel_size=5, activation='relu'))
# 	# model.add(MaxPool1D(strides=8))
# 	# model.add(Conv1D(filters=50, kernel_size=5, activation='relu'))
# 	# model.add(MaxPool1D(strides=8))
# 	# model.add(Flatten())
# 	# model.add(Dense(units=1, activation='softmax'))
# 	model = Sequential()
# 	model.add(Dense(60, input_dim=20000, activation='relu')) #make input_dim = 5000/4000 :(
# 	model.add(Dense(24, activation='relu'))
# 	model.add(Dense(8, activation='relu'))
# 	model.add(Dense(4, activation='sigmoid'))
 	
#  	# multiclass :::categorical_crossentropy

# 	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 	model.fit(data_x, dummy_y, batch_size=32, epochs=5)

# 	model.save('classify_model.h5') 


	############################TESTING

	f_test_x = []
	f_test_y = []
	for j in range((2*temp.shape[0])/3,temp.shape[0]):
		test_x = []
		test_y = []
		for i in range(1, temp.shape[1] - 1):
			try: 
				print PATH+ temp.iloc[j, 0].split('.')[0] +'.wav'
				data, sampling_rate = librosa.load(PATH + temp.iloc[j, 0].split('.')[0] +'.wav', sr=44100 )
				temp_data = data[int(temp.iloc[j, i]):int(temp.iloc[j, i+1])]
				temp_label = temp.iloc[j, 0].split('__')[0]
#NOT USED YET
#				(mfccs, chroma, mel, contrast, tonnetz) = extract_feature(temp_data,sampling_rate)
				test_x.append(temp_data)
				test_y.append(temp_label)
			except:
				pass
		if len(test_y) >0:
			f_test_y.append(test_y)
			f_test_x.append(test_x)


	print f_test_x,f_test_y
	for i in range(0,len(f_test_x)):
		f_test_x[i] = pad_sequences(f_test_x[i], maxlen=20000, dtype='float', padding='post', truncating='post', value=0.)
		f_test_x[i] = f_test_x[i] / np.max(f_test_x[i])
	# data_y.append('normal')
	# data_y.append('extrahls')
	# data_y.append('artifact')
	# data_y.append('murmur')
	

	#data_x = data_x[:,:,np.newaxis]
	for i in range(0,len(f_test_y)):
		f_test_y[i] = pd.Series(f_test_y[i])
		f_test_y[i].value_counts()
		f_test_y[i] = f_test_y[i].map({'/normal':0, '/extrahls':1,'/artifact':2,'/murmur':3}).values






	get_files_in_directory(f_test_x,f_test_y,model)
	# print model








model = load_model('classify_model.h5')
read_segment('segment_seta.csv',model)
print model
