# 	Mentor:Dr. Mustaq Ahmed
#	Akshay Arora,Anand Mukut Tirkey, Harivamshi Valamkonda

from sklearn import svm
import csv
from sklearn.externals import joblib


#X represents input of the file
#Y represents output labels

X=[]
Y=[]

#loading data from CSV files to X and Y lists

with open("vector.csv","rb") as csv_file:
	csv_reader=csv.reader(csv_file)
	for row in csv_reader:
		print row
		Z=(row[1][1:len(row[1])-1].split(' '))
		while '' in Z: Z.remove('')
		for i in range(0,len(Z)):
		
			Z[i]=Z[i].strip()
			
		temp=[]
		print Z
		for v in Z:
			temp.append(float(v))
		X.append(temp)
		Y.append((int(row[2])+1)/2)

print(X)
print(Y)


## library used for svm :- sklearn
## kernels tested :- rbf,poly(3),linear,sigmoid 



clf = svm.SVC(kernel= 'poly')
info=clf.fit(X, Y)
print info

joblib.dump(clf, 'svm_model.pkl') 
