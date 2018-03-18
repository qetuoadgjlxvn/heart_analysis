# 	Mentor:Dr. Mustaq Ahmed
#	Akshay Arora,Anand Mukut Tirkey, Harivamshi Valamkonda


import csv
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
#import librosa
import numpy as np

names=['a','b','c','d','e']


#database list stores all the audiofile names 
#along with their labels 
database=[]

#vector_database list stores tuples of size three
#	0--- file name
#  	1--- vector of the audio size
#	2--- class label 

vector_database=[]
X=[]
Y=[]
for c in names:
	with open('training/training-'+c+'/REFERENCE.csv','rb') as csv_file:
		csv_reader=csv.reader(csv_file)
		for row in csv_reader:
			database.append(row)
		

cnt=11

#extract features for particular sound file
#uses only mfcc as a feature 

def convert_to_vector(filename):
	(rate,sig) = wav.read(filename)
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	#print(fbank_feat)
	print("######################")
	vector1=(fbank_feat[1:3,:][1])
	#print(vector1)
	vector2=(fbank_feat[1:3,:][0])
	
	#print vector2
	print("######################")
	z=np.hstack((vector1,vector2))
#	vector.extend(list(fbank_feat[1:3,:][1]))
	return z


for tuples in database:
	
	#print (fbank_feat[1:3,:])
	vector=convert_to_vector("training/training-"+tuples[0][0]+"/"+tuples[0]+".wav")
	print(tuples[0],vector)
	vector_database.append([tuples[0],vector,tuples[1]])
	cnt+=1
	if(cnt==10):
		break

#print vector_database
# saving the vector_database as csv file
with open('vector.csv', 'w') as csvfile:
	fieldnames = ['file_name', 'vector','class']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	#writer.writeheader()
	for elem in vector_database:
		writer.writerow({'file_name': elem[0], 'vector': elem[1], 'class':elem[2]})




