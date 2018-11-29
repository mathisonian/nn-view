

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, HTML


class PyTorchModel(object):
  def __init__(self):
    pass

  def __repr__(self):
    return 'I am the representation of a pytorch model'

  def __str__(self):
    return 'I am a pytorch model'

  def display_diagram(self):
    display(HTML("<b>:) okay?</b>"))

