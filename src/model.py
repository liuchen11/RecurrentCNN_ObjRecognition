import numpy as np

import theano
import theano.tensor as T

from libparser import *
from hiddenLayer import *
from convLayer import *
from logisticRegression import *
from recurrentConvLayer import *

class model(object):

	def __init__(self,learn_rate,n_epochs,filters,times,batch_size):
		'''
		>>>type learn_rate: float
		>>>para learn_rate: learning rate

		>>>type n_epochs: int
		>>>para n_epochs: maximum epochs of the iteration

		>>>type filters: list
		>>>para filters: the num of filters in each layer

		>>>type times: list
		>>>para times: the num of recurrent times in each layer

		>>>type batch_size: int
		>>>para batch_size: num of instances in each batch
		'''
		rng=np.random.RandomState(2011010539)
		self.learn_rate=learn_rate
		self.n_epochs=n_epochs
		self.filters=filters
		self.batch_size=batch_size

		self.x=T.dmatrix('x')			#inputs
		self.y=T.lvector('y')			#labels
		self.lr=T.dscalar('learning_rate')	#learning rate

		input=self.x.reshape((batch_size,3,32,32))

		self.layer0=ConvPool(
			rng=rng,
			input=input,
			shape=[batch_size,3,32,32],
			filters=[filters[0],3,5,5],
			pool=[2,2],
			dropout=True
		)

		self.layer1=RecurrentConvLayer(
			rng=rng,
			input=self.layer0.output,
			shape=[batch_size,filters[0],14,14],
			filters=[filters[1],filters[0],3,3],
			rfilter=[filters[1],filters[1],3,3],
			alpha=0.001,beta=0.75,
			N=filters[1]/8+1,
			time=times[1],
			pool=[1,1]
		)

		self.layer2=RecurrentConvLayer(
			rng=rng,
			input=self.layer1.output,
			shape=[batch_size,filters[1],12,12],
			filters=[filters[2],filters[1],3,3],
			rfilter=[filters[2],filters[2],3,3],
			alpha=0.001,beta=0.75,
			N=filters[2]/8+1,
			time=times[2],
			pool=[2,2]
		)

		self.layer3=RecurrentConvLayer(
			rng=rng,
			input=self.layer2.output,
			shape=[batch_size,filters[2],5,5],
			filters=[filters[3],filters[2],3,3],
			rfilter=[filters[3],filters[2],3,3],
			alpha=0.001,beta=0.75,
			N=filters[3]/8+1,
			time=times[3],
			pool=[1,1]
		)

		self.layer4=RecurrentConvLayer(
			rng=rng,
			input=self.layer3.output,
			shape=[batch_size,filters[3],3,3],
			filters=[filters[4],filters[3],3,3],
			rfilter=[filters[4],filters[3],3,3],
			alpha=0.001,beta=0.75,
			N=filters[4]/8+1,
			time=times[4],
			pool=[1,1]
		)

		self.layer5=LogisticRegression(
			input=self.layer4.output.flatten(2),
			n_in=filters[4],
			n_out=10
		)

		self.cost=self.layer5.negative_log_likelyhood(self.y)
		self.error=self.layer5.errors(self.y)

		self.params=self.layer0.param+self.layer1.param+self.layer2.param+self.layer3.param+self.layer4.param+self.layer5.param
		#self.params=self.layer0.param+self.layer1.param+self.layer2.param+self.layer3.param+self.layer5.param
		self.grads=T.grad(self.cost,self.params)

		self.updates=[
			(param_i,param_i-self.lr*grad_i)
			for param_i,grad_i in zip(self.params,self.grads)
		]
		print 'construction completed!'


	def train_validate_test(self,train_set_x,train_set_y,validate_set_x,validate_set_y,test_set_x,test_set_y):
		'''
		>>>add validate set to avoid overfitting
		>>>type train_set_x/validate_set_x/test_set_x: T.dmatrix
		>>>para train_set_x/validate_set_x/test_set_x: data of instances of training/validate/test set
		>>>type train_set_y/validate_set_y/test_set_y: T.ivector
		>>>para train_set_y/validate_set_y/test_set_y: labels of instances of training/validate/test set
		'''
		assert train_set_x.shape[0]==train_set_y.shape[0]
		assert validate_set_x.shape[0]==validate_set_y.shape[0]
		assert test_set_x.shape[0]==test_set_y.shape[0]

		trainX=theano.shared(train_set_x,borrow=True)
		trainY=theano.shared(train_set_y,borrow=True)
		validateX=theano.shared(validate_set_x,borrow=True)
		validateY=theano.shared(validate_set_y,borrow=True)
		testX=theano.shared(test_set_x,borrow=True)
		testY=theano.shared(test_set_y,borrow=True)

		train_set_batches=train_set_y.shape[0]/self.batch_size
		validate_set_batches=validate_set_y.shape[0]/self.batch_size
		test_set_batches=test_set_y.shape[0]/self.batch_size

		index=T.iscalar('index')
		lr=T.dscalar('lr')

		train_model=theano.function(
			[index,lr],
			self.cost,
			updates=self.updates,
			givens={
			self.x:trainX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:trainY[index*self.batch_size:(index+1)*self.batch_size],
			self.lr:lr
			}
			)
		validate_model=theano.function(
			[index],
			self.error,
			givens={
			self.x:validateX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:validateY[index*self.batch_size:(index+1)*self.batch_size]
			}
			)
		test_model=theano.function(
			[index],
			1.0-self.error,
			givens={
			self.x:testX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:testY[index*self.batch_size:(index+1)*self.batch_size]
			}
			)

		max_iter=10000
		max_iter_increase=2
		improvement_threshold=1.0
		validate_fre=min(train_set_batches,max_iter/2)

		best_validation_loss=np.inf
		best_iter=0
		test_score_mean=0.0

		epoch=0
		done_looping=False

		rate=self.learn_rate
		min_error=1.0
		present_error=1.0

		while (epoch<self.n_epochs) and (not done_looping):
			epoch+=1
			if present_error>0.5:
				rate=0.01
			elif present_error>0.35:
				rate=0.005
			elif present_error>0.3:
				rate=0.003
			elif present_error>0.25:
				rate=0.002
			else:
				rate=0.001

			for batch_index in xrange(train_set_batches):
				iter_num=(epoch-1)*train_set_batches+batch_index
				#print self.layer0.w.get_value()[0,0,0,0]
				if iter_num%100==0:
					print 'training@iter=%d/%d'%(iter_num,train_set_batches*self.n_epochs)
				
				cost_now=train_model(batch_index,rate)

				if (iter_num+1)%validate_fre==0:
					validation_losses=[
					validate_model(i)
					for i in xrange(validate_set_batches)
					]
					validation_loss_mean=np.mean(validation_losses)
					print 'epoch %i, batch_index %i/%i, validation accuracy %f %%'%(epoch,batch_index+1,train_set_batches,(1.0-validation_loss_mean)*100.)
					print 'best result:%f %%'%((1.0-min_error)*100)

					if validation_loss_mean<best_validation_loss:
						if validation_loss_mean<best_validation_loss*improvement_threshold:
							max_iter=max(max_iter,iter_num*max_iter_increase)

						best_validation_loss=validation_loss_mean
						best_iter=iter_num
						stop_optimal=0

						test_scores=[
						test_model(i)
						for i in xrange(test_set_batches)
						]
						test_score_mean=np.mean(test_scores)
						print '\nepoch %i, batch_index %i/%i, test accuracy %f %%'%(epoch,batch_index+1,train_set_batches,test_score_mean*100.)

						present_error=1-test_score_mean
						if present_error<min_error:
							min_error=present_error
						

				if iter_num>=max_iter:
					done_looping=True
					break

		print 'best validate accuracy of %f %% at iteration %i, with test accuracy %f %%'%((1.0-best_validation_loss)*100.,best_iter+1,test_score_mean*100.)

