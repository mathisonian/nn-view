from collections import OrderedDict
import numpy as np
import math



from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML
from NNView.layers.layer import Layer
from vdom.helpers import h1, p, img, div, b
from vdom.svg import svg, circle, rect, g, text, line
import keras


min_conv_size = 20
max_conv_size = 150

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
                layer = Layer(layer_type, input_shape, output_shape, activation, channels, kernel, stride)
                formatted.append(layer)
            elif ('Conv1D' in layer_type or 'Conv2D' in layer_type or 'Conv3D' in layer_type):
                input_shape = l.input.get_shape().as_list()[1]
                output_shape = None
                channels = l.output.get_shape().as_list()[3]
                kernel = l.__dict__['kernel_size'][0]
                stride = l.__dict__['strides'][0]
                layer = Layer(layer_type, input_shape, output_shape, activation, channels, kernel, stride)
                formatted.append(layer)
            elif ('InputLayer' in layer_type):
                pass
            else: 
                # raise Exception(f"Layer type not supported: {layer_type}")
                pass
            
            i+=1

        return formatted



    def __repr__(self):

        '''
        Do two passes to find the maximum layer size, and scale everything else based on that.
        '''

        max_input_size = 0
        max_linear_size = 0
        max_channel_size = 0
        max_layer_height = 0

        filter_offset_size = 5

        for layer in self.trace:
            if 'Conv' in layer.type:

                if layer.channels > max_channel_size:
                    max_channel_size = layer.channels

                if layer.input_shape > max_input_size:
                    max_input_size = layer.input_shape
            else:
                # print(layer.input_shape)
                if layer.input_shape > max_linear_size:
                    max_linear_size = layer.input_shape

        channel_factor = 1 if max_channel_size == 0 else math.log(max_channel_size)

        print(channel_factor)

        layer_offset = 0
        vertical_offset = 0
        svg_width = 0
        svg_height = 500
        conv_group = []
        annotation_group = []
        linear_group = []
        linear_layer_width = 50
        max_group_height = None

        last_layer = None
        last_offset = 0

        for layer in self.trace:

            # print(layer.type, layer.input_shape, layer.output_shape, layer.channels)
            if layer.type == 'Linear':
                vertical_offset = 0
                padding = 30
                max_height = max_group_height if max_group_height is not None else svg_height - 2 * padding
                rect_height = np.interp(layer.input_shape, [0, max_linear_size], [0, max_height])
                effective_width = (math.sin(math.radians(45)) * rect_height)
                effective_height = (math.cos(math.radians(45)) * rect_height)
                layer.effective_height = effective_height
                layer.effective_width = effective_width
                rect_top = np.interp(rect_height, [0, max_height], [svg_height / 2, (svg_height - max_height) / 2])
                layer_offset += min(rect_height / 4, 30)
                if max_group_height is None:
                    layer_offset += effective_width / 2


                if last_layer is not None and 'Linear' in last_layer.type:
                    annotation_group.append(g(
                        line(
                        x1=str(layer_offset - effective_width / 2 + 20),
                        y1=str(svg_height / 2 - effective_height / 2 + 5),
                        x2=str(last_offset - last_layer.effective_width / 2 + 25),
                        y2=str(svg_height / 2 - last_layer.effective_height / 2 + 5),
                        stroke="black",
                        strokeWidth="2"
                        )
                    ))
                    annotation_group.append(g(
                        line(
                        x1=str(layer_offset + effective_width / 2 + 15),
                        y1=str(svg_height / 2 + effective_height / 2 + 5),
                        x2=str(last_offset + last_layer.effective_width / 2 + 20),
                        y2=str(svg_height / 2 + last_layer.effective_height / 2),
                        stroke="black",
                        strokeWidth="2"
                        )
                    ))
                elif last_layer is not None and 'Conv' in last_layer.type:
                    annotation_group.append(
                        line(
                        x1=str(last_offset + last_layer.rect_width),
                        y1=str(last_layer.vertical_offset),
                        x2=str(layer_offset - effective_width / 2 + 20),
                        y2=str(svg_height / 2 - effective_height / 2 + 5),
                        stroke='#222'
                        )
                    )
                    annotation_group.append(
                        line(
                        x1=str(last_offset + last_layer.rect_width + (last_layer.channels - 1) * filter_offset_size),
                        y1=str(last_layer.vertical_offset + (last_layer.channels - 1) * filter_offset_size + last_layer.rect_width),
                        x2=str(layer_offset + effective_width / 2 + 15),
                        y2=str(svg_height / 2 + effective_height / 2 + 5),
                        stroke='#222'
                        )
                    )

                linear_group.append(g(
                [rect(
                    x=str((linear_layer_width - 20) / 2),
                    y=str(rect_top),
                    width='10',
                    height=str(rect_height),
                    fill='#efefef',
                    stroke='#bbb',
                    strokeWidth="3",
                    transform='rotate(-45, ' + str(linear_layer_width / 2) + ', ' + str(svg_height / 2) +')'
                ),
                text(
                    str(layer.output_shape),
                    x=str(effective_width / 2 + rect_height / 8),
                    y=str(svg_height / 2 + effective_height / 2 - rect_height / 8),
                    textAnchor="start",
                    fill="black"
                ),
                text(
                    str(layer.input_shape),
                    x=str(-effective_width / 2 ),
                    y=str(svg_height / 2 - effective_height / 2 + rect_height / 8),
                    textAnchor="end",
                    fill="black"
                )
                ],
                transform='translate(' + str(layer_offset) + ', 0)'
                ))


                last_offset = layer_offset
                layer_offset += (effective_width / 2)

            elif 'Conv' in layer.type:
                layer._channels = layer.channels
                layer.channels = round(layer.channels / channel_factor)
                rect_width = np.interp(layer.input_shape, [0, max_input_size], [min_conv_size, max_conv_size])
                group_height = rect_width + filter_offset_size * layer.channels
                if max_group_height is None or group_height > max_group_height:
                    max_group_height = group_height
                vertical_offset = np.interp(group_height, [0, svg_height], [svg_height / 2, 0])
                layer.vertical_offset = vertical_offset
                layer.rect_width = rect_width
                rects = []
                # print(group_height)
                for i in range(layer.channels):
                    rects.append(
                        rect(
                        x=str(i * filter_offset_size),
                        y=str(i * filter_offset_size),
                        width=str(rect_width),
                        height=str(rect_width),
                        fill='#dddddd' if i % 2 == 0 else '#bbbbbb')
                        )

                rects.append(
                    rect(
                        x=str((layer.channels - 1) * filter_offset_size + rect_width / 4),
                        y=str((layer.channels - 1) * filter_offset_size + rect_width / 4),
                        width=str(np.interp(layer.kernel_size, [0, max_input_size], [min_conv_size, max_conv_size])),
                        height=str(np.interp(layer.kernel_size, [0, max_input_size], [min_conv_size, max_conv_size])),
                        fill='none',
                        stroke='#222'
                    )
                )
                rects.append(
                    text(
                        str(layer._channels) + ' @ ' + str(layer.input_shape) + ' X ' + str(layer.input_shape),
                        x=str((rect_width / 2 + layer.channels * filter_offset_size) / 2),
                        y=str(group_height + 40),
                        textAnchor="middle",
                        fill="black"
                    )
                )

                if last_layer and 'Conv' in last_layer.type:
                    annotation_group.append(
                        line(
                        x1=str(last_offset + (last_layer.channels - 1) * filter_offset_size + last_layer.rect_width / 4 + (np.interp(last_layer.kernel_size, [0, max_input_size], [min_conv_size, max_conv_size]))),
                        y1=str(last_layer.vertical_offset + (last_layer.channels - 1) * filter_offset_size + last_layer.rect_width / 4),
                        x2=str(layer_offset + (layer.channels - 1) * filter_offset_size + rect_width / 2),
                        y2=str(vertical_offset + (layer.channels - 1) * filter_offset_size + rect_width / 2),
                        stroke='#222'
                        )
                    )
                    annotation_group.append(
                        line(
                        x1=str(last_offset + (last_layer.channels - 1) * filter_offset_size + last_layer.rect_width / 4 + (np.interp(last_layer.kernel_size, [0, max_input_size], [min_conv_size, max_conv_size]))),
                        y1=str(last_layer.vertical_offset + (last_layer.channels - 1) * filter_offset_size + last_layer.rect_width / 4 + (np.interp(last_layer.kernel_size, [0, max_input_size], [min_conv_size, max_conv_size]))),
                        x2=str(layer_offset + (layer.channels - 1) * filter_offset_size + rect_width / 2),
                        y2=str(vertical_offset + (layer.channels - 1) * filter_offset_size + rect_width / 2),
                        stroke='#222'
                        )
                    )

                conv_group.append(g(rects, transform='translate(' + str(layer_offset) + ', ' + str(vertical_offset) + ')'))
                last_offset = layer_offset
                layer_offset += rect_width + layer.channels * filter_offset_size + 20

            else:
                print(layer.type)
                print('???')
                continue

            last_layer = layer

        groups = g(conv_group + annotation_group + linear_group)
        display(svg(
            groups,
            width=str(1.10 * layer_offset),
            height=str(svg_height),
            style={ 'overflow': 'visible', 'max-width': 'none', 'height': 'auto'}
        ))

        return ''

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