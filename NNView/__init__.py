from os.path import join, dirname
from IPython.display import display, HTML

with open(join(dirname(__file__), 'VERSION')) as f:
    __version__ = f.read().strip()

from . import models


import codecs


location = join(dirname(__file__), 'lib/nnview.js')
js = codecs.open(location, "r", "utf-8").read()
display(HTML('<script>' + js + '</script>'))



# display(HTML('<script>window.a = 1234; console.log("b")</script>'))