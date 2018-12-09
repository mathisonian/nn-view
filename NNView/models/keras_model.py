from collections import OrderedDict

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML
from NNView.layers.layer import Layer
import keras


class KerasModel(object):
    def __init__(self, net):
        self.network = net
        self.trace = self.format_trace(self.trace_net(self.network))

  # @Returns list of layers in the network in a formatted way
    def trace_net(self, net):
        return net.layers

    def format_trace(self, trace):
        formatted = []

        i = 0
        while i < len(trace):
            l = trace[i]
            input_shape = None
            output_shape = None
            activation = None
            channels = None
            kernel = None
            stride = None

            layer_type = l.__class__.__name__


             # Usually Conv layer, batch norm, then activation 
            # check if layer has an activation function
            if (i+2 < len(trace) and isinstance(trace[i+2], keras.layers.core.Activation)):
                activation = trace[i+2].__dict__['activation'].__name__
                i += 2 # skip the activation function and batch norm

            if ('Linear' in layer_type):
                assert (l.input.get_shape().as_list()[1] == l.input.get_shape().as_list()[2]) # Check that the input is square
                input_shape = l.input.get_shape().as_list()[1]
                assert (l.output.get_shape().as_list()[1] == l.output.get_shape().as_list()[2]) # Check that the input is square
                output_shape = l.output.get_shape().as_list()[1]
            elif ('Conv1D' in layer_type or 'Conv2D' in layer_type or 'Conv3D' in layer_type):
                input_shape = l.input.get_shape().as_list()[1]
                output_shape = None
                channels = l.output.get_shape().as_list()[3]
                kernel = l.__dict__['kernel_size'][0]
                stride = l.__dict__['strides'][0]
            elif ('InputLayer' in layer_type):
                pass
            else: 
                # raise Exception(f"Layer type not supported: {layer_type}")
                pass

            layer = Layer(layer_type, input_shape, output_shape, activation, channels, kernel, stride)
            formatted.append(layer)
            
            i+=1

        return formatted



    def __repr__(self):
      return 'I am the representation of a keras model' 
    def __str__(self):
      return 'I am a keras model'   
    def display_diagram(self):
      # All data that is needed (input dim, output dim, kernel size, num of featres)
      display(HTML("<b>:) okay?</b>"))  
    def summarize_trace(self):
      # Sumamrize
      for i, layer in enumerate(self.trace):
          print(f"Layer {i}")
          # print("-" * 10)
          print("Type of layer: ", layer['layer_type'])
          print(f"Input dimensions: {layer['input_dim']} x {layer['input_dim']}")
          print(f"Num features: {layer['num_features']}")
          print(f"Kernel size: {layer['kernel_size']} x {layer['kernel_size']}")
          print(f"Output dimensions: {layer['output_dim']} x {layer['output_dim']}")
          print("-" * 10)
    
## TESTING 1
# import tensorflow
# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np

# model = ResNet50(weights='imagenet')
# ktm = KerasModel(model)
# import pdb;pdb.set_trace()

## TESTING 2
# Adapted from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Create first network with Keras
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy
# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# # load dataset
# dataset = numpy.loadtxt("../data/data.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # create model
# model = Sequential()
# model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# model.add(Dense(8, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='sigmoid'))
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# # calculate predictions
# predictions = model.predict(X)
# # round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)