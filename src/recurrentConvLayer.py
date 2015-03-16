import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

def ReLU(x):
	return theano.tensor.switch(x<0,0,x)

def Padding(input,width,height):
	'''
	>>>type input: T.matrix
	>>>para input: input tensor

	>>>type width/height: int
	>>>para width/height: the width/height of the filter
	'''
	input_width=input.shape[0]
	input_height=input.shape[1]

	pad_width_left=(width+1)/2
	pad_width_right=width/2
	pad_height_up=(height+1)/2
	pad_height_down=height/2

	left=T.zeros([pad_width_left,input_height])
	right=T.zeros([pad_width_right,input_height])
	input=T.concatenate([left,input,right],axis=0)

	up=T.zeros([width+input_width,pad_height_up])
	down=T.zeros([width+input_width,pad_height_down])
	input=T.concatenate([up,input,down],axis=1)
	return input

class RecurrentConvLayer(object):
	
	def __init__(self,rng,input,shape,filters,rfilter,alpha,beta,N,time,pool):
		'''
		>>>type rng: numpy.random.RandomState
		>>>para rng: random seed
		
		>>>type input: T.tensor4
		>>>para input: input data

		>>>type shape: tuple or list of length 4
		>>>para shape: (batch_size,num of input feature maps, image height, image width)

		>>>type filters: tuple or list of length 4
		>>>para filters: (num of filters, num of input feature maps, filter height, filter width)

		>>>type rfilter: tuple or list of length 3
		>>>para rfilter: (num of filters, recurrent filter height, recurrent filter width)

		>>>type alpha,beta,N: int or float
		>>>para alpha,beta,N: used in the formulation of recurent state

		>>>type time: int
		>>>para time: the num of iteration in the recurrent layer

		>>>type pool: tuple or list of length 2
		>>>para pool: pooling size
		'''

		assert shape[1]==filters[1]
		assert filters[0]=rfilter[0]
		self.input=input
		layer_size=(shape[0],filters[0],shape[2]-filters[2]+1,shape[3]-filters[3]+1)
		
		inflow=np.prod(filters[1:])
		outflow=filters[0]*np.prod(filters[2:])/np.prod(pool)

		w_bound=np.sqrt(6./(inflow+outflow))
		rw_bound=np.sqrt(3./np.prod(rfilter))

		w_in_init=np.asarray(rng.uniform(low=-w_bound,high=w_bound,size=filters),dtype=theano.config.floatX)
		self.w_in=theano.shared(value=w_in_init,name='w_in')
		w_r_init=np.asarray(rng.uniform(low=-rw_bound,high=rw_bound,size=rfilter),dtype=theano.config.floatX)
		self.w_r=theano.shared(value=w_r_init,name='w_r')

		b_init=np.zeros(shape=filters[0],dtype=theano.config.floatX)
		self.b=theano.shared(value=b_init,name='b_in')
		b_r_init=np.zeros(shape=layer_size,dtype=theano.config.floatX)
		self.b_r=theano.shared(value=b_r_init,name='b_r')

		conv_input=conv.conv2d(
			input=input,
			filters=self.w_in,
			filter_shape=filters,
			image_shape=shape
			)

		def step(x_input,state):
			tmp_value=T.zeros(state.shape)
			for i in xrange(layer_size[0]):
				for j in xrange(layer_size[1]):
					padded_input=Padding(input=state[i,j],height=rfilter[1],width=rfilter[2])
					conv_recurrent=conv.conv2d(
						input=padded_input.dimshuffle('x','x',0,1),
						filters=self.w_r[j].dimshuffle('x','x',0,1),		#warning non-shared variable!
						filter_shape=[1,1,rfilter[1],rfilter[2]],
						image_shape=[1,1,layer_size[2],layer_size[3]]
					)
					tmp_value=T.set_subtensor(tmp_value[i,j],ReLU(conv_recurrent[0,0]+x_input[i,j]))
				for x in xrange(layer_size[2]):
					for y in xrange(layer_size[3]):
						for k in xrange(layer_size[1]):
							norm=0.0
							for n in xrange(N):
								if k-N/2+n>=0 and k-N/2+n<layer_size[1]:
									norm+=tmp_value[i,k-N/2+n,x,y]**2
							norm=(norm*alpha/N+1)**beta
							state=T.set_subtensor(state[i,k,x,y],tmp_value[i,k,x,y]/norm)
			return state

		state,_=theano.scan(step,sequences=conv_input,output_info=self.b_r,n_step=time)

		pool_out=downsample.max_pool_2d(
			input=state,
			ds=pool,
			ignore_border=True
		)
		self.output=pool_out+self.b.dimshuffle('x',0,'x','x')
		self.param=[self.w_in,self.w_r,self.b,self.b_r]
		
