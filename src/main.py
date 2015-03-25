import sys
import theano
import numpy as np

from libparser import *
from convLayer import *
from hiddenLayer import *
from logisticRegression import *
from recurrentConvLayer import *
from model import *

sys.setrecursionlimit(3000)

path='../data/'

if __name__=='__main__':
	train=unzip(path+'train_data')
	validate=unzip(path+'validate_data')
	test=unzip(path+'test_data')

	train_data=np.asarray(train['data'],dtype=theano.config.floatX)
	train_label=np.asarray(train['labels'],dtype=theano.config.floatX)
	validate_data=np.asarray(validate['data'],dtype=theano.config.floatX)
	validate_label=np.asarray(validate['labels'],dtype=theano.config.floatX)
	test_data=np.asarray(test['data'],dtype=theano.config.floatX)
	test_label=np.asarray(test['labels'],dtype=theano.config.floatX)

	print 'data loaded!'

	machine=model(
		learn_rate=0.005,
		n_epochs=100,
		filters=[96,96,96,96,96],
		times=[0,1,1,1,1],
		batch_size=25
		)

	print 'pre-process the data...'
	
	train_set_data=np.asarray(train_data,dtype=theano.config.floatX)
	train_set_label=np.asarray(train_label,dtype=int)
	validate_set_data=np.asarray(validate_data,dtype=theano.config.floatX)
	validate_set_label=np.asarray(validate_label,dtype=int)
	test_set_data=np.asarray(test_data,dtype=theano.config.floatX)
	test_set_label=np.asarray(test_label,dtype=int)

	for i in xrange(len(train_set_data)):
		train_set_data[i]-=train_set_data[i].mean()

	for i in xrange(len(validate_set_data)):
		validate_set_data[i]-=validate_set_data[i].mean()

	for i in xrange(len(test_set_data)):
		test_set_data[i]-=test_set_data[i].mean()

	print 'start training the model...'

	machine.train_validate_test(
		train_set_data,train_set_label,
		validate_set_data,validate_set_label,
		test_set_data,test_set_label
		)
