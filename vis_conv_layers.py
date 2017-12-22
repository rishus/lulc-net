#REAME: This code should be run after learn_keras_4.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras import backend as K
from keras.utils import np_utils

#https://github.com/yashk2810/Visualization-of-Convolutional-Layers/blob/master/Visualizing%20Filters%20Python3%20Theano%20Backend.ipynb
def layer_to_visualize(layer, figname):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)
    
    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))
    # Visualization of each filter of the layer
    #fig = plt.figure(figsize=(12,8))
    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)
#    figname = str(layer) + '.png'
    for i in range(len(convolutions)):
        #figname = str(i) + '.png'
        ax = fig.add_subplot(n,n,i+1)
	ax.axis('off')
        ax.imshow(convolutions[i], cmap='gray')
    fig.savefig(figname, format='png')



