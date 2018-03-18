# this is a fun project
import csv
from pydub import AudioSegment
import os


def doYourJob(parent_dir, sub_dirs):
	print '###', len(sub_dirs)
	filename_label = []
	for i in range(0,len(sub_dirs)):
		sub_dir = sub_dirs[i]
        print sub_dir
        with open(parent_dir + sub_dirs[i] + 'REFERENCE.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
            	print row
                filename_label.append(row)
        for fn in filename_label:
            try:
                print fn
                if fn[1] == '1':
                	print('/home/user/Desktop/project_heartt/'+parent_dir+sub_dirs[i]+fn[0]+'.wav')
                	sound = AudioSegment.from_wav('/home/user/Desktop/project_heartt/'+parent_dir+sub_dirs[i]+fn[0]+'.wav')
                	halfway_point = len(sound) // 2
                	first_half = sound[:halfway_point]
                	second_half = sound[halfway_point:]
                	print('/home/user/Desktop/project_heartt/'+parent_dir+sub_dirs[i]+fn[0]+'-1'+'.wav')
                	first_half.export('/home/user/Desktop/project_heartt/'+parent_dir+sub_dirs[i]+fn[0]+'-1'+'.wav',format = "wav")
                	second_half.export('/home/user/Desktop/project_heartt/'+parent_dir+sub_dirs[i]+fn[0]+'-2'+'.wav',format = "wav")
                	print('opening', parent_dir + sub_dirs[i] + 'REFERENCE.csv')
                	with open (parent_dir + sub_dirs[i] + 'REFERENCE.csv', 'ab') as csvf:
                		writer = csv.writer(csvf)
                		writer.writerow([fn[0]+'-1']+['1'])
                		writer.writerow([fn[0]+'-2']+['1'])
                else:
                	print('inside else')
            except Exception, e:
                print 'Error encountered while parsing file: ', fn
                continue

parent_dir = 'data/'
tr_sub_dirs = ['training-f/']
#don't call below function otherwise it will create more partition of already partitioned files
#doYourJob(parent_dir,tr_sub_dirs)
sound = AudioSegment.from_wav('/home/user/Desktop/project_heartt/data/training-a/a0001.wav')

halfway_point = len(sound) // 2
first_half = sound[:halfway_point]
first_half.export('/home/user/Desktop/demo.wav',format = "wav")