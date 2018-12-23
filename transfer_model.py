
from keras import backend as K
from keras.models import load_model
from progress.bar import Bar # sudo pip install progress


def transferModel(model_old, model_new, parameters=None, mfile=None, sparseLayersList=[]):

    # model refers to the old correct and complete model as gained through training in the current epoch
    # s2s_sparseomodel_newnlycorrect refers to a new model created from scratch that only contains 
    # correct dense encoderlayers that were set in the transformer class
    # now we have to set the rest
    print('Transfering model...')

    bar = Bar('Processing', max=len(model_new.layers), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

    count = 0
    for layer in model_new.layers:
        # print(layer.name+'   '+model.layers[count].name)
        if layer.name not in sparseLayersList:
            weightsfromlastepoch = model_old.layers[count].get_weights()
            layer.set_weights(weightsfromlastepoch)
        count = count + 1
        bar.next()
    bar.finish()

    return model_new