import os, sys
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
from sparsity import *
from keras import backend as K
from transfer_model import transferModel
from helper import *
import pickle

################## parameters ####################
maxepoches = 30
epsilon = 20 # control the sparsity level as discussed in the paper
zeta = 0.3 # the fraction of the weights removed
d_model = 512
d_inner_hid = 512
layers = 2
n_head = 8
d_k=64
d_v=64
len_limit=70
dropout=0.1
batch_size=64
max_len=120
###################################################

model_parameters = {'maxepoches': maxepoches, 'epsilon':epsilon, 'zeta':zeta, 'd_model':d_model, 'd_inner_hid':d_inner_hid, 'layers':layers, 'n_head':n_head, 'd_k':d_k, 'd_v':d_v, 'len_limit':len_limit, 'dropout':dropout, 'batch_size':batch_size, 'max_len':max_len}
filepath = createHistoryFile(model_parameters, sys.argv)

############### Load trainingsdata ################
if 'testdata' in sys.argv:
	itokens, otokens = dd.MakeS2SDict('data/test_subset/en2de.s2s.txt', dict_file='data/test_subset/en2de_word.txt')
	Xtrain, Ytrain = dd.MakeS2SData('data/test_subset/en2de.s2s.txt', itokens, otokens, h5_file='data/test_subset/en2de.h5')
	Xvalid, Yvalid = dd.MakeS2SData('data/test_subset/en2de.s2s.valid.txt', itokens, otokens, h5_file='data/test_subset/en2de.valid.h5')

if 'origdata' in sys.argv:
	itokens, otokens = dd.MakeS2SDict('data/en2de.s2s.txt', dict_file='data/en2de_word.txt')
	Xtrain, Ytrain = dd.MakeS2SData('data/en2de.s2s.txt', itokens, otokens, h5_file='data/en2de.h5')
	Xvalid, Yvalid = dd.MakeS2SData('data/en2de.s2s.valid.txt', itokens, otokens, h5_file='data/en2de.valid.h5')
	# gen = dd.S2SDataGenerator('data/en2de.s2s.txt', itokens, otokens, batch_size=batch_size, max_len=max_len)

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

###################################################


from transformer import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

#Define optimizer
adam = Adam(0.001, 0.9, 0.98, epsilon=1e-9)

#path to saved model
mfile = 'models/en2de.model.h5'

################ callbacks ################
lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
# lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
###########################################

if 'sparse' in sys.argv:
	
	initParams = initSparseWeights(epsilon, n_head=n_head, d_k=d_k, d_v=d_v, layers=layers)
	s2s = Transformer(itokens, otokens, len_limit=len_limit, d_model=d_model, d_inner_hid=d_inner_hid, \
				n_head=n_head, d_k=d_k, d_v=d_v, layers=layers, dropout=dropout, weightsForSparsity=initParams)
	s2s.compile(adam)
	s2s.model.summary()

	if 'load_existing_model' in sys.argv:
		s2s.model.summary()
		try: s2s.model.load_weights(mfile)
		except: print('\n\nnew model')
	else:
		print('*** New model ***')

	for epoch in range(0,maxepoches):
		print('epoch #'+str(epoch))

		history = s2s.model.fit([Xtrain, Ytrain], None, batch_size=batch_size, epochs=1, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver], verbose=1)

		writeEpochHistoryToDisk(history, filepath)

		parameters = weightsEvolution(s2s, initParams=initParams, zeta=zeta, layers=layers)

		# create new Transformer with sparse layers and masked weights
		s2s_sparseonlycorrect = Transformer(itokens, otokens, len_limit=len_limit, d_model=d_model, d_inner_hid=d_inner_hid, \
					n_head=n_head, d_k=d_k, d_v=d_v, layers=layers, dropout=dropout, weightsForSparsity=parameters)
		# compile the model
		s2s_sparseonlycorrect.compile(adam)
		
		# get the list of layers to ignore
		sparseLayersList = getSparseLayersList()

		# transfer into new model
		s2s.model = transferModel(model_old=s2s.model, model_new=s2s_sparseonlycorrect.model, parameters=parameters, mfile=mfile, sparseLayersList=sparseLayersList)
		
		# compile this new model again
		s2s.compile(adam)

elif 'originalWithTransfer' in sys.argv:

	s2s = Transformer(itokens, otokens, len_limit=len_limit, d_model=d_model, d_inner_hid=d_inner_hid, \
				n_head=n_head, d_k=d_k, d_v=d_v, layers=layers, dropout=dropout)
	s2s.compile(adam)
	s2s.model.summary()

	if 'load_existing_model' in sys.argv:
		s2s.model.summary()
		try: s2s.model.load_weights(mfile)
		except: print('\n\nnew model')
	else:
		print('*** New model ***')

	for epoch in range(0,maxepoches):
		print('epoch #'+str(epoch))

		history = s2s.model.fit([Xtrain, Ytrain], None, batch_size=batch_size, epochs=1, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])

		writeEpochHistoryToDisk(history, filepath)

		# create new Transformer with sparse layers and masked weights
		s2s_new = Transformer(itokens, otokens, len_limit=len_limit, d_model=d_model, d_inner_hid=d_inner_hid, \
					n_head=n_head, d_k=d_k, d_v=d_v, layers=layers, dropout=dropout)
		# compile the model
		s2s_new.compile(adam)

		# transfer into new model, which is in this case an empty Transformer model to proof transfer correctness
		s2s.model = transferModel(model_old=s2s.model, model_new=s2s_new.model)
		
		# compile this new model again
		s2s.compile(adam)

elif 'originalImplementation' in sys.argv:

	s2s = Transformer(itokens, otokens, len_limit=len_limit, d_model=d_model, d_inner_hid=d_inner_hid, \
				n_head=n_head, d_k=d_k, d_v=d_v, layers=layers, dropout=dropout)
	s2s.compile(adam)
	s2s.model.summary()

	history = s2s.model.fit([Xtrain, Ytrain], None, batch_size=batch_size, epochs=maxepoches, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])

	writeEpochHistoryToDisk(history, filepath)


