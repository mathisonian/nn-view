import torchvision.models as models
import torch
import torch.nn as nn
import types

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML
from vdom.helpers import h1, p, img, div, b
from vdom.svg import svg, circle, rect
from NNView.layers.layer import Layer


LAYER_TYPES = ['Linear', 'Conv']
LAYER_TYPES_FEATURES = [('in_features', 'out_features'), ('in_channels', '')]

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

    return svg(
      circle(r=10, cx=20, cy=30, fill='#ff0000'),
      width=500,
      height=300
    )
    # display(
    #   div(
    #     svg(
    #         circle(r=10, cx=20, cy=30, fill='#ff0000'),
    #         width=500,
    #         height=300
    #     )
    #   )
    # )
    # return ''
    # return display(
    #   div(
    #     svg(
    #       circle(r=10, cx=20, cy=30, fill='#ff0000'),
    #       rect(width=30, height=30, x=200, y=300, fill='#0000ff'),
    #       width=500,
    #       height=300
    #     )
    #   )
    # )

  def __str__(self):
    return self.trace_net

  # Monkey patching to store network info in list called @param trace
  def _mk_wrapper(self, trace, super_method):
    def _wrapper(self, *args, **kwargs):
        output = super_method(*args, **kwargs)
        trace.append(self)
        # trace.append((self, args, kwargs, output))
        return output
    return _wrapper

  # Binds newly defined wrapper to network layers so that we can re-create  layer order and input/output
  # @Returns a list where each element is a tuple:
  def trace_net(self):
    trace = []
    for child in net.children():
        old_call = child.forward
        new_method = self._mk_wrapper(trace, old_call)
        child.forward = types.MethodType(new_method, child)
    return trace 

  # @param trace is list of raw layer output
  def format_trace(self, trace):
      formatted = []

      # for i,l in enumerate(trace):
      i = 0
      while i < len(trace):
        l = trace[i]
        layer_type = l.__class__.__name__
        input_shape = None
        output_shape = None
        activation = None
        kernel = None
        stride = None

        # check if next layer is an activation function 
        if (i+1 < len(trace) and isinstance(trace[i+1], torch.nn.modules.activation.Threshold)):
            activation = trace[i+1].__class__.__name__
            i += 1 # skip the activation function
        
        if ('Linear' in layer_type): 
            input_shape = l.__dict__['in_features']
            output_shape = l.__dict__['out_features']
        elif ('Conv' in layer_type): # Allows for Conv1d, Conv2d, Conv3d
            input_shape = l.__dict__['in_channels']
            output_shape = l.__dict__['out_channels']
            kernel = l.__dict__['kernel_size'][0] # without [0], returns tuple - e.g., (3,)
            stride = l.__dict__['stride'][0] # without [0], returns tuple - e.g., (3,)
            
        else: 
          raise Exception(f"Layer type not supported: {layer_type}")

        layer = Layer(layer_type, input_shape, output_shape, activation, kernel, stride)
        formatted.append(layer)
        # import pdb; pdb.set_trace()
        i+=1
      return formatted
    

  def display_diagram(self):
    self.trace

    _id = 'graph-' + str(id(self))
    js = 'lenet = LeNet("#' + _id + '"); console.log(lenet); lenet.redraw({ architecture: [{ numberOfSquares: 8, squareWidth: 128, stride: 8 }, { numberOfSquares: 8, squareWidth: 64, stride: 16 }, { numberOfSquares: 24, squareWidth: 48, stride: 8 }, { numberOfSquares: 24, squareWidth: 16, stride: 8 }], architecture2: [256, 128] }); lenet.redistribute({betweenLayers_: [40, 10, -20, -20] })'

    display(HTML('<div><div id="' + _id + '"></div><script>' + js + '</script></div>'))

  def summarize_trace(self):
    # Sumamrize
    for i, (net, args, kwargs, output) in enumerate(self.trace):
        print(f"Layer {i}:")
        # print("-" * 10)
        print("Network: ", net)
        print("Parameters: ", [p.shape for p in net.parameters()])
        print("Arg Dimesions: ", [x.shape for x in args])
        print("Output Dimesions: ", output.shape)
        print("-" * 10)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.c1 = nn.Conv1d(16, 33, 3, stride=2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.c1(torch.randn(20, 16, 50))
        return out

input_size = 784
hidden_size = 500
num_classes = 10

net = Net(input_size, hidden_size, num_classes)
net_input = torch.randn(input_size)
ptm = PyTorchModel(net, net_input)

## TRY for nn.Sequential type
import pdb; pdb.set_trace()