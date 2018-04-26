#
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
import os
import csv


#PATH='/home/user/Desktop/heart_sound_segmentation/heartbeat-sounds/sounds_aif/Atraining_normal/Atraining_normal'
PATH='/home/user/Desktop/heart_sound_segmentation/heartbeat-sounds/set_a'
def train_model():
	temp = pd.read_csv('Atraining_normal_seg.csv')
	temp.head()

	print temp.head()

	data, sampling_rate = librosa.load(PATH+'/normal__201102081321.wav', sr=44100 )
	#display.waveplot(data, sr=sampling_rate)

	# plt.subplot(3, 1, 2)
	# display.waveplot(data, sr=sampling_rate)
	# plt.title('Stereo')
	# plt.show()



	data_x = []
	data_y = []
	for j in range(temp.shape[0]):
		for i in range(1, temp.shape[1] - 1):
			try: 
				print PATH+ '/normal__' +temp.iloc[j, 0].split('.')[0] +'.wav'
				data, sampling_rate = librosa.load(PATH+ '/normal__' +temp.iloc[j, 0].split('.')[0] +'.wav', sr=44100 )
				temp_data = data[int(temp.iloc[j, i])-2000:int(temp.iloc[j, i])+2000]
				temp_label = temp.iloc[:, i].name.split('.')[0]
				data_x.append(temp_data)
				data_y.append(temp_label)
			except:
				pass




	data_x = pad_sequences(data_x, maxlen=5000, dtype='float', padding='post', truncating='post', value=0.)
	data_x = data_x / np.max(data_x)

	#data_x = data_x[:,:,np.newaxis]
	data_y = pd.Series(data_y)
	data_y.value_counts()

	data_y = data_y.map({'S1':0, 'S2':1}).values

	print data_x,'##############',data_y
	# model = Sequential()

	# model.add(InputLayer(input_shape=data_x.shape[1:]))

	# model.add(Conv1D(filters=50, kernel_size=5, activation='relu'))
	# model.add(MaxPool1D(strides=8))
	# model.add(Conv1D(filters=50, kernel_size=5, activation='relu'))
	# model.add(MaxPool1D(strides=8))
	# model.add(Flatten())
	# model.add(Dense(units=1, activation='softmax'))
	model = Sequential()
	model.add(Dense(60, input_dim=5000, activation='relu'))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))


	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	model.fit(data_x, data_y, batch_size=32, epochs=5)

	model.save('segment_model.h5') 
	print model


def segment(filename,model):

	data, sampling_rate = librosa.load(PATH+filename, sr=44100 )
	# plt.subplot(3, 1, 2)
	# display.waveplot(data, sr=sampling_rate)
	# plt.show()
	print len(data)
	i=0
	data_x=[]
	idx=[]
	# while(i < len(data)):
	# 	for j in range(i+5000,i+15000,1000):
	# 		temp_x=data[i:j]
	# 		idx.append((i,j))
	# 		print i," # ",j
	# 		data_x.append(temp_x)

	# 	i+=5000

	while (i < len(data)):
		j = i + 4000
		temp_x= data[i:j]
		idx.append((i,j))
		data_x.append(temp_x)
		i += 500

	data_x = pad_sequences(data_x, maxlen=5000, dtype='float', padding='post', truncating='post', value=0.)
	data_x = data_x / np.max(data_x)

	s1_s2_list=[]
	s1_s2_list.append(filename)
	flag=0
	z= model.predict(data_x)
	for i in range(0,len(idx)):
		if z[i] <= 0.2 and flag==0:
			print "s1" , idx[i]
			s1_s2_list.append((idx[i][0]+idx[i][1])/2)
			flag=1
		if z[i]>=0.7 and flag==1:
			print "s2" ,idx[i]
			s1_s2_list.append((idx[i][0]+idx[i][1])/2)
			flag=0
#		print idx[i],"##",z[i]
	return s1_s2_list



def get_files_in_directory(directory,model):
	a=[]
	max_len=0
	for filename in os.listdir(directory):
		print filename,"###########"
		x=(segment('/'+filename,model))
		if len(x)>max_len:
			max_len=len(x)
		a.append(x)

	for i in range(0,len(a)):
		while(len(a[i])<max_len):
			a[i].append('')

	print a
	### PENDING TO WRITE IN FILES
	with open("segment_seta.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(a)



#train_model()

model = load_model('segment_model.h5')
get_files_in_directory(PATH,model)
print model

#plot_model(model, to_file='model.png')
print model.layers[0].get_weights()
