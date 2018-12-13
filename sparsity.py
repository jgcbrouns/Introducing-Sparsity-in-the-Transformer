from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
import h5py
import time
from progress.bar import Bar # sudo pip install progress

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


def getSparseLayersList():
    sparseLayersList = []
    for i in range(0,6):
        sparseLayersList.append('sparse_qs_'+str(i))
        sparseLayersList.append('sparse_ks_'+str(i))
        sparseLayersList.append('sparse_vs_'+str(i))
    return sparseLayersList



def transferModel(model, model_sparseonlycorrect, parameters, mfile, sparseLayersList):

    # model refers to the old correct and complete model as gained through training in the current epoch
    # s2s_sparseomodel_sparseonlycorrectnlycorrect refers to a new model created from scratch that only contains 
    # correct dense encoderlayers that were set in the transformer class
    # now we have to set the rest
    print('Transfering model...')

    bar = Bar('Processing', max=len(model_sparseonlycorrect.layer), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

    count = 0
    for layer in model_sparseonlycorrect.layer:
        # print(layer.name+'   '+model.layers[count].name)
        if layer.name not in sparseLayersList:
            weightsfromlastepoch = model.layers[count].get_weights()
            layer.set_weights(weightsfromlastepoch)
        count = count + 1
        bar.next()
    bar.finish()

    return model_sparseonlycorrect



def weightsEvolution(self, initParams, zeta, layers):
    # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
    
    parameters = {}
    
    #for every stacked encoder (or decoder) layer
    for i in range(0, layers):

        #w1, w2, w3
        sparse_qs_w = self.model.get_layer('sparse_qs_'+str(i)).get_weights()
        sparse_ks_w = self.model.get_layer('sparse_ks_'+str(i)).get_weights()
        sparse_vs_w = self.model.get_layer('sparse_vs_'+str(i)).get_weights()

        noPar_sparse_qs = initParams['noPar_sparse_qs_'+str(i)]
        noPar_sparse_ks = initParams['noPar_sparse_ks_'+str(i)]
        noPar_sparse_vs = initParams['noPar_sparse_vs_'+str(i)]

        #wm1, wm1core
        [sparse_qs_wm, sparse_qs_wCore] = rewireMask(self, sparse_qs_w[0], noPar_sparse_qs, zeta=zeta)
        [sparse_ks_wm, sparse_ks_wCore] = rewireMask(self, sparse_ks_w[0], noPar_sparse_ks, zeta=zeta)
        [sparse_vs_wm, sparse_vs_wCore] = rewireMask(self, sparse_vs_w[0], noPar_sparse_vs, zeta=zeta)

        sparse_qs_w[0] = sparse_qs_w[0] * sparse_qs_wCore
        sparse_ks_w[0] = sparse_ks_w[0] * sparse_ks_wCore
        sparse_vs_w[0] = sparse_vs_w[0] * sparse_vs_wCore

        parameters['noPar_sparse_qs_'+str(i)] = noPar_sparse_qs
        parameters['noPar_sparse_ks_'+str(i)] = noPar_sparse_ks
        parameters['noPar_sparse_vs_'+str(i)] = noPar_sparse_vs

        parameters['sparse_qs_wm_'+str(i)] = sparse_qs_wm
        parameters['sparse_ks_wm_'+str(i)] = sparse_ks_wm
        parameters['sparse_vs_wm_'+str(i)] = sparse_vs_wm

        parameters['sparse_qs_w_'+str(i)] = sparse_qs_w
        parameters['sparse_ks_w_'+str(i)] = sparse_ks_w
        parameters['sparse_vs_w_'+str(i)] = sparse_vs_w

    return parameters


def initSparseWeights(epsilon, n_head, d_k, d_v, layers):
	# generate an Erdos Renyi sparse weights mask for each layer
    initParams = {}

    for i in range(0, layers):
        
        [noPar_sparse_qs, sparse_qs_wm] = createWeightsMask(epsilon,512, n_head*d_k)
        [noPar_sparse_ks, sparse_ks_wm] = createWeightsMask(epsilon,512, n_head*d_k)
        [noPar_sparse_vs, sparse_vs_wm] = createWeightsMask(epsilon,512, n_head*d_v)

        initParams['noPar_sparse_qs_'+str(i)] = noPar_sparse_qs
        initParams['noPar_sparse_ks_'+str(i)] = noPar_sparse_ks
        initParams['noPar_sparse_vs_'+str(i)] = noPar_sparse_vs

        initParams['sparse_qs_wm_'+str(i)] = sparse_qs_wm
        initParams['sparse_ks_wm_'+str(i)] = sparse_ks_wm
        initParams['sparse_vs_wm_'+str(i)] = sparse_vs_wm

        initParams['sparse_qs_w_'+str(i)] = None
        initParams['sparse_ks_w_'+str(i)] = None
        initParams['sparse_vs_w_'+str(i)] = None
        
    return initParams


# def replace_intermediate_layer_in_keras(model, layer_id, new_layer, layer_name='NONAME'):
#     from keras.models import Model

#     layers = [l for l in model.layers]

#     x = layers[0].output
#     for i in range(1, len(layers)):
#         print(layers[i].name)
#         if layers[i].name == layer_name:
#             x = new_layer(x)
#         else:
#             x = layers[i](x)

#     new_model = Model(input=layers[0].input, output=x)
#     return new_model