# 	Mentor:Dr. Mustaq Ahmed
#	Akshay Arora,Anand Mukut Tirkey, Harivamshi Valamkonda

import csv
#from vector_convert import convert_to_vector
from sklearn.externals import joblib
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

database=[]

#extract features for particular sound file
#uses only mfcc as a feature 

def convert_to_vector(filename):
	(rate,sig) = wav.read(filename)
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	vector=list(fbank_feat[1:3,:][0])
	vector.extend(list(fbank_feat[1:3,:][1]))
	return vector

with open('validation/REFERENCE.csv','rb') as csv_file:
	csv_reader=csv.reader(csv_file)
	for row in csv_reader:
		database.append(row)
cnt=0
result=0

#loading pre trained model from pickle
#file

clf = joblib.load('svm_model.pkl') 
for tuples in database:
	#print (fbank_feat[1:3,:])
	
	vector=convert_to_vector("validation/"+tuples[0]+".wav")

	#using model to predict the class
	result_class=clf.predict([vector])
	print tuples[0],(int(tuples[1])+1)/2,result_class
	if((int(tuples[1])+1)/2) == result_class[0]:
		result+=1
	cnt+=1
	if(cnt==100):
		break

print result
