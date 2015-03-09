import logging

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax

logger=logging.getLogger(__name__)

mode=theano.Mode(linker='cvm')

class RNN(object):
	'''
	Implement of recurrent neural network with theano
	'''

	def __init__(self,input,n_in,n_hidden,n_out,activation):
		'''
		>>>type input: theano.tensorType
		>>>para input: input data

		>>>type n_in,n_hidden,n_out: int
		>>>para n_in,n_hidden,n_out: the number of neurons in the input layer, hidden layer and output layer

		>>>tyoe activation: T.elemwise.Elemwise
		>>>para activation: activate function
		'''
		self.input=input
		self.activation=activation
		
		W_in_init=np.asarray(np.random.uniform(size=(n_in,n_hidden),high=0.05,low=-0.05),dtype=theano.config.floatX)
		self.W_in=theano.shared(value=W_in_init,name='W_in')

		W_r_init=np.asarray(np.random.uniform(size=(n_hidden,n_hidden),high=0.05,low=-0.05),dtype=theano.config.floatX)
		self.W_r=theano.shared(value=W_r_init,name='W_recurrent')

		W_out_init=np.asarray(np.random.uniform(size=(n_hidden,n_out),high=0.05,low=-0.05),dtype=theano.config.floatX)
		self.W_out=theano.shared(value=W_out_init,name='W_out')

		b_in_init=np.asarray(np.random.uniform(size=(n_hidden,),high=0.05,low=-0.05),dtype=theano.config.floatX)
		self.b_in=theano.shared(value=b_in_init,name='b_in')

		b_r_init=np.asarray(np.random.uniform(size=(n_hidden,),high=0.05,low=-0.05),dtype=theano.config.floatX)
		self.b_r=theano.shared(value=b_r_init,name='b_r')

		b_out_init=np.asarray(np.random.uniform(size=(n_out,),high=0.05,low=-0.05),dtype=theano.config.floatX)
		self.b_out=theano.shared(value=b_out_init,name='b_out')

		self.softmax=softmax
		self.params=[self.W_in,self.W_r,self.W_out,self.b_in,self.b_r,self.b_out]
		
		self.update={}
		for param in self.params:
			upd=np.zeros(shape=param.get_value(borrow=True).shape,dtype=theano.config.floatX)
			self.update[param]=theano.shared(value=upd,name='upd')
			

		def step(x,h):
			new_h=self.activation(T.dot(x,self.W_in)+T.dot(h,self.W_r)+self.b_r)
			new_output=T.dot(new_h,self.W_out)+self.b_out
			return new_h,new_output

		[self.hidden,self.y_pred],_=theano.scan(step,sequences=self.input,outputs_info=[self.b_in,None])

		self.pYgivenX=softmax(self.y_pred)
		self.predict=T.argmax(self.pYgivenX,axis=-1)
		
	def loss(self,y):
		'''
		>>>loss function
		>>>type y: list
		>>>para y: teaching signals
		'''

		return -T.mean(T.log(self.pYgivenX)[T.arange(y.shape[0]),y])

	def errors(self,y):
		'''
		>>>calculate error rate
		>>>type y: list
		>>>para y: teaching signals
		'''
		return -T.mean(T.neq(self.predict,y))

if __name__=='__main__':
	'''Test basic RNN'''
	input=np.asarray([[3],[4],[5],[6]],dtype=theano.config.floatX)
	net=RNN(input,1,1,1,T.tanh)
