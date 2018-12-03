import torchvision.models as models
import torch
import torch.nn as nn
import types

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML
from vdom.helpers import h1, p, img, div, b
from vdom.svg import svg, circle, rect

class PyTorchModel(object):
  def __init__(self, net, net_input):
    self.network = net
    self.input = net_input

    # Override forward pass and get layer order and info
    self.trace = self.trace_net(self.network)
    self.network(self.input)

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
        trace.append((self, args, kwargs, output))
        return output
    return _wrapper
  # Binds newly defined wrapper to network layers so that we can re-create  layer order and input/output
  # @Returns a list where each element is a tuple:

  def trace_net(self, net):
    trace = []
    for child in net.children():
        old_call = child.forward
        new_method = self._mk_wrapper(trace, old_call)
        child.forward = types.MethodType(new_method, child)
    return trace 

  def format_trace(self):
    pass

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
