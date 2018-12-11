import os, sys
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
from sparsity import *



itokens, otokens = dd.MakeS2SDict('data/test_subset/en2de.s2s.txt', dict_file='data/test_subset/en2de_word.txt')
Xtrain, Ytrain = dd.MakeS2SData('data/test_subset/en2de.s2s.txt', itokens, otokens, h5_file='data/test_subset/en2de.h5')
Xvalid, Yvalid = dd.MakeS2SData('data/test_subset/en2de.s2s.valid.txt', itokens, otokens, h5_file='data/test_subset/en2de.valid.h5')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

from transformer import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

maxepoches = 3
# set model parameters
epsilon = 20 # control the sparsity level as discussed in the paper
zeta = 0.3 # the fraction of the weights removed


d_model = 512
# s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
# 				   n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)

# mfile = 'models/en2de.model.h5'

# lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
# #lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
# model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

# s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
# s2s.model.summary()
# try: s2s.model.load_weights(mfile)
# except: print('\n\nnew model')

if 'test' in sys.argv:
	print(s2s.decode_sequence_fast('A black dog eats food .'.split(), delimiter=' '))
	while True:
		quest = input('> ')
		print(s2s.decode_sequence_fast(quest.split(), delimiter=' '))
		rets = s2s.beam_search(quest.split(), delimiter=' ')
		for x, y in rets: print(x, y)
elif 'original' in sys.argv:
	s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])
elif 'sparse' in sys.argv:
	initParams = initSparseWeights(epsilon, n_head=8, d_k=64, d_v=64)
	s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
				   n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1, weightsForSparsity=initParams)
	
	mfile = 'models/en2de.model.h5'

	lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
	#lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
	model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
	s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
	s2s.model.summary()
	try: s2s.model.load_weights(mfile)
	except: print('\n\nnew model')
	
	for epoch in range(0,maxepoches):
		print('epoch #'+str(epoch))

		adam = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
		s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))

		s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=1, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])

		#ugly hack to avoid tensorflow memory increase for multiple fit_generator calls. Theano shall work more nicely this but it is outdated in general
		parameters = weightsEvolution(s2s, initParams=initParams, zeta=zeta)
		K.clear_session()
		s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
					n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1, weightsForSparsity=parameters)
		s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))

	# self.accuracies_per_epoch=np.asarray(self.accuracies_per_epoch)

