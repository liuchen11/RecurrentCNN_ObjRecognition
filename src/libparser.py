import cPickle
import cv2
import numpy as np
import os
import sys

classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','struck']

def unzip(filename):
	fo=open(filename,'rb')
	data=cPickle.load(fo)
	return data

def draw(data, label):
	if len(data)!=3072:
		print 'Wrong Pixel Size: 3072 Required, %d Given' % len(data)
		return
	pixels=np.zeros((32,32,3),np.uint8);
	for i in range(0,3072):
		p1=i%32
		p2=(i%1024)/32
		p3=i/1024
		pixels[p1,p2,p3]=data[i]
	cv2.imwrite(classes[label]+'.bmp',pixels)

if __name__=='__main__':
	if len(sys.argv)<2:
		print 'Usage python libparser.py <inputfile>'
		exit(0)
	filename=sys.argv[1]
	data=unzip(filename)
	index=-1
	while True:
		print 'input the index number of the picture (0-9999) or minus to exit'
		query=raw_input('>>>')
		if query=='' or query==None:
			index+=1
		else:
			index=int(query)
			if index<0:
				break
			if index>=10000:
				index=9999
		draw(data['data'][index],data['labels'][index])
	clear=raw_input('clear the output file?(Y/N)')
	if clear.lower()=='y' or clear.lower()=='yes':
		files=os.listdir('.')
		for f in files:
			name=f[:f.find('.')]
			if name in classes:
				os.remove(f)