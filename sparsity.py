#sudo ssh -i ~/.ssh/google_compute_engine jeroenbrouns@104.199.49.176

from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras import backend as K
from keras.utils import np_utils

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}

def createWeightsMask(epsilon,noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print ("Create Sparse Matrix: No parameters, NoRows, NoCols ",noParameters,noRows,noCols)
    return [noParameters,mask_weights]

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def rewireMask(self,weights, noWeights, zeta):
    # rewire weight matrix

    # remove zeta largest negative and smallest positive weights
    values = np.sort(weights.ravel())
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)
    largestNegative = values[int((1-zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos +zeta * (values.shape[0] - lastZeroPos)))]
    rewiredWeights = weights.copy()
    rewiredWeights[rewiredWeights > smallestPositive] = 1
    rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    weightMaskCore = rewiredWeights.copy()

    # add zeta random weights
    nrAdd = 0
    noRewires = noWeights - np.sum(rewiredWeights)
    while (nrAdd <= noRewires):
        i = np.random.randint(0, rewiredWeights.shape[0])
        j = np.random.randint(0, rewiredWeights.shape[1])
        if (rewiredWeights[i, j] == 0):
            rewiredWeights[i, j] = 1
            nrAdd += 1

    return [rewiredWeights, weightMaskCore]

def weightsEvolution(self, initParams, zeta):
    # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
    
    #w1, w2, w3
    sparse_qs_0_w = self.model.get_layer("sparse_qs_0").get_weights()
    sparse_ks_0_w = self.model.get_layer("sparse_ks_0").get_weights()
    sparse_vs_0_w = self.model.get_layer("sparse_vs_0").get_weights()

    noPar_sparse_qs_0 = initParams['noPar_sparse_qs_0']
    noPar_sparse_ks_0 = initParams['noPar_sparse_ks_0']
    noPar_sparse_vs_0 = initParams['noPar_sparse_vs_0']

    #wm1, wm1core
    [sparse_qs_0_wm, sparse_qs_0_wCore] = rewireMask(self, sparse_qs_0_w[0], noPar_sparse_qs_0, zeta=zeta)
    [sparse_ks_0_wm, sparse_ks_0_wCore] = rewireMask(self, sparse_ks_0_w[0], noPar_sparse_ks_0, zeta=zeta)
    [sparse_vs_0_wm, sparse_vs_0_wCore] = rewireMask(self, sparse_vs_0_w[0], noPar_sparse_vs_0, zeta=zeta)

    sparse_qs_0_w[0] = sparse_qs_0_w[0] * sparse_qs_0_wCore
    sparse_ks_0_w[0] = sparse_ks_0_w[0] * sparse_ks_0_wCore
    sparse_vs_0_w[0] = sparse_vs_0_w[0] * sparse_vs_0_wCore

    parameters = {
		'noPar_sparse_qs_0' : noPar_sparse_qs_0,
		'noPar_sparse_ks_0' : noPar_sparse_ks_0,
		'noPar_sparse_vs_0' : noPar_sparse_vs_0,
		'sparse_qs_0_wm' : sparse_qs_0_wm,
		'sparse_ks_0_wm' : sparse_ks_0_wm,
		'sparse_vs_0_wm' : sparse_vs_0_wm,
		'sparse_qs_0_w' : sparse_qs_0_w,
		'sparse_ks_0_w' : sparse_ks_0_w,
		'sparse_vs_0_w' : sparse_vs_0_w
	}

    return parameters; 


def initSparseWeights(epsilon, n_head, d_k, d_v):
	# generate an Erdos Renyi sparse weights mask for each layer
	[noPar_sparse_qs_0, sparse_qs_0_wm] = createWeightsMask(epsilon,512, n_head*d_k)
	[noPar_sparse_ks_0, sparse_ks_0_wm] = createWeightsMask(epsilon,512, n_head*d_k)
	[noPar_sparse_vs_0, sparse_vs_0_wm] = createWeightsMask(epsilon,512, n_head*d_v)

	initParams = {
		'noPar_sparse_qs_0' : noPar_sparse_qs_0,
		'noPar_sparse_ks_0' : noPar_sparse_ks_0,
		'noPar_sparse_vs_0' : noPar_sparse_vs_0,
		'sparse_qs_0_wm' : sparse_qs_0_wm,
		'sparse_ks_0_wm' : sparse_ks_0_wm,
		'sparse_vs_0_wm' : sparse_vs_0_wm,
		'sparse_qs_0_w' : None,
		'sparse_ks_0_w' : None,
		'sparse_vs_0_w' : None
	}

	return initParams