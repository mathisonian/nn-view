from collections import OrderedDict

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML


class KerasModel(object):
  def __init__(self, net):
    self.network = net
    self.trace = self.trace_net(self.network)

  # @Returns list of layers in the network in a formatted way
  def trace_net(self, net):
    trace = []
    network_layers = net.layers

    for layer in network_layers:
        # import pdb;pdb.set_trace()
        layer_type = layer.__class__.__name__
        # import pdb;pdb.set_trace()
        try:
            assert (layer.input.get_shape().as_list()[1] == layer.input.get_shape().as_list()[2]) # Check that the input is square
            input_dim = layer.input.get_shape().as_list()[1]
        except:
            pass

        try:
            num_features = layer.filters
        except:
            num_features = -1 # layer doesn't have filters

        try:
            assert (layer.kernel_size[0] == layer.kernel_size[1]) # Check that filter is square
            kernel_size = layer.kernel_size[0]
        except:
            kernel_size = -1 # layer doesn't have kernel

        # May not need the output dimensions
        try:
            assert (layer.output.get_shape().as_list()[1] == layer.output.get_shape().as_list()[2]) # Check that the output is square
            output_dim = layer.output.get_shape().as_list()[1]
        except:
            pass

        layer_info = {
            # 'name': name,
            'layer_type': layer_type,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'num_features': num_features,
            'kernel_size': kernel_size
        }

        trace.append(layer_info)

    return trace

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