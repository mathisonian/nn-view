

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML
from vdom.helpers import h1, p, img, div, b
from vdom.svg import svg, circle, rect


class PyTorchModel(object):
  def __init__(self):
    pass

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
    return 'I am a pytorch model'

  def display_diagram(self):

    _id = 'graph-' + str(id(self))
    js = 'lenet = LeNet("#' + _id + '"); console.log(lenet); lenet.redraw({ architecture: [{ numberOfSquares: 8, squareWidth: 128, stride: 8 }, { numberOfSquares: 8, squareWidth: 64, stride: 16 }, { numberOfSquares: 24, squareWidth: 48, stride: 8 }, { numberOfSquares: 24, squareWidth: 16, stride: 8 }], architecture2: [256, 128] }); lenet.redistribute({betweenLayers_: [40, 10, -20, -20] })'

    display(HTML('<div><div id="' + _id + '"></div><script>' + js + '</script></div>'))
