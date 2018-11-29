from setuptools import setup
from setuptools import find_packages
from os.path import join, dirname
# We need io.open() (Python 3's default open) to specify file encodings
import io

with open(join(dirname(__file__), 'NNView/VERSION')) as f:
    version = f.read().strip()

try:
    # obtain long description from README
    # Specify encoding to get a unicode type in Python 2 and a str in Python 3
    readme_path = join(dirname(__file__), 'README.md')
    with io.open(readme_path, encoding='utf-8') as fr:
        README = fr.read()
except IOError:
    README = ''


install_requires = [
]

tests_require = [
]

setup(
    name="NNView",
    version=version,
    description="Quickly generate architecture diagrams of PyTorch or Keras models in a notebook.",
    long_description=README,
    classifiers=[
    ],
    keywords="",
    author="",
    author_email="",
    url="",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
    },
)