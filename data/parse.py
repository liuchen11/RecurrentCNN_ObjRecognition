import cPickle

fo1=open('data_batch_1','rb')
fo2=open('data_batch_2','rb')
fo3=open('data_batch_3','rb')
fo4=open('data_batch_4','rb')
fo5=open('data_batch_5','rb')
fo6=open('test_batch','rb')

p1=cPickle.load(fo1)
p2=cPickle.load(fo2)
p3=cPickle.load(fo3)
p4=cPickle.load(fo4)
p5=cPickle.load(fo5)
p6=cPickle.load(fo6)

train={'data':[],'labels':[]}
validate={'data':[],'labels':[]}
test={'data':[],'labels':[]}

train['data'][00000:10000]=p1['data']
train['data'][10000:20000]=p2['data']
train['data'][20000:30000]=p3['data']
train['data'][30000:40000]=p4['data']
train['data'][40000:50000]=p5['data']
train['labels']=p1['labels']+p2['labels']+p3['labels']+p4['labels']+p5['labels']
validate['data']=p5['data']
validate['labels']=p5['labels']
test['data']=p6['data']
test['labels']=p6['labels']

trainfile=open('train_data','wb')
validatefile=open('validate_data','wb')
testfile=open('test_data','wb')

cPickle.dump(train,trainfile, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(validate,validatefile, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(test,testfile, protocol=cPickle.HIGHEST_PROTOCOL)
