import torchvision.models as models
import torch
import torch.nn as nn
import types
import numpy as np
import math

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML
from vdom.helpers import h1, p, img, div, b
from vdom.svg import svg, circle, rect, g, text, line
from NNView.layers.layer import Layer


LAYER_TYPES = ['Linear', 'Conv']
LAYER_TYPES_FEATURES = [('in_features', 'out_features'), ('in_channels', '')]

min_conv_size = 20
max_conv_size = 150

class PyTorchModel(object):
  def __init__(self, net, net_input):
    self.network = net
    self.input = net_input

    # Override forward pass and get layer order and info
    # self.trace = self.trace_net(self.network)
    self.trace = self.trace_net()

    self.network(self.input) # run forward pass of networks
    self.trace = self.format_trace(self.trace)

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
    return self.trace_net

  # Monkey patching to store network info in list called @param trace
  def _mk_wrapper(self, trace, super_method):
    def _wrapper(self, *args, **kwargs):
        output = super_method(*args, **kwargs)
        trace.append([self, args])
        # import pdb; pdb.set_trace()
        # trace.append((self, args, kwargs, output))
        return output
    return _wrapper

  # Binds newly defined wrapper to network layers so that we can re-create  layer order and input/output
  # @Returns a list where each element is a tuple:
  def trace_net(self):
    trace = []
    def check_children(arr):
      for child in arr:
          if 'Sequential' in child.__class__.__name__ or 'Block' in child.__class__.__name__ or '_DenseLayer' in child.__class__.__name__:
            check_children(child.children())
          else:
            old_call = child.forward
            new_method = self._mk_wrapper(trace, old_call)
            child.forward = types.MethodType(new_method, child)

    check_children(self.network.children())
    return trace

  # @param trace is list of raw layer output
  def format_trace(self, trace):
      layer_info = [t[0] for t in trace]
      size_info = [t[1][0].shape for t in trace]

      formatted = []

      # for i,l in enumerate(trace):
      i = 0

      while i < len(layer_info):
        l = layer_info[i]
        s = size_info[i]
        layer_type = l.__class__.__name__
        input_shape = None
        output_shape = None
        channels = None
        activation = None
        kernel = None
        stride = None

        # check if next layer is an activation function
        if (i+1 < len(trace) and isinstance(trace[i+1][0], torch.nn.modules.activation.Threshold)):
            activation = trace[i+1][0].__class__.__name__
            i += 1 # skip the activation function

        if ('Linear' in layer_type):

            input_shape = s[-1]
            output_shape = l.__dict__['out_features']
            channels = None
        elif ('Conv' in layer_type): # Allows for Conv1d, Conv2d, Conv3d

            input_shape = s[2]
            output_shape = None
            channels = l.__dict__['out_channels']
            kernel = l.__dict__['kernel_size'][0] # without [0], returns tuple - e.g., (3,)
            stride = l.__dict__['stride'][0] # without [0], returns tuple - e.g., (3,)

        else:
          # print(f"Layer type not supported: {layer_type}")
          i += 1
          continue
          # raise Exception(f"Layer type not supported: {layer_type}")

        layer = Layer(layer_type, input_shape, output_shape, activation, channels, kernel, stride)
        formatted.append(layer)
        # import pdb; pdb.set_trace()
        i+=1
      return formatted


  # def display_diagram(self):
  #   self.trace

  #   _id = 'graph-' + str(id(self))
  #   js = 'lenet = LeNet("#' + _id + '"); console.log(lenet); lenet.redraw({ architecture: [{ numberOfSquares: 8, squareWidth: 128, stride: 8 }, { numberOfSquares: 8, squareWidth: 64, stride: 16 }, { numberOfSquares: 24, squareWidth: 48, stride: 8 }, { numberOfSquares: 24, squareWidth: 16, stride: 8 }], architecture2: [256, 128] }); lenet.redistribute({betweenLayers_: [40, 10, -20, -20] })'

  #   display(HTML('<div><div id="' + _id + '"></div><script>' + js + '</script></div>'))

  # def summarize_trace(self):
  #   # Sumamrize
  #   for i, (net, args, kwargs, output) in enumerate(self.trace):
  #       print(f"Layer {i}:")
  #       # print("-" * 10)
  #       print("Network: ", net)
  #       print("Parameters: ", [p.shape for p in net.parameters()])
  #       print("Arg Dimesions: ", [x.shape for x in args])
  #       print("Output Dimesions: ", output.shape)
  #       print("-" * 10)



## TESTING
# class Net(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.c1 = nn.Conv1d(16, 33, 3, stride=2)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.c1(torch.randn(20, 16, 50))
#         return out

# input_size = 784
# hidden_size = 500
# num_classes = 10

# net = Net(input_size, hidden_size, num_classes)
# net_input = torch.randn(input_size)
# ptm = PyTorchModel(net, net_input)

# ## TRY for nn.Sequential type
# import pdb; pdb.set_trace()


## TESTING 2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


# net = Net()
# net_input = torch.randn(1, 1, 32, 32)
# ptm = PyTorchModel(net, net_input)

## TRY for nn.Sequential type
# import pdb; pdb.set_trace()