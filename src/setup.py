#!/usr/bin/env python
from setuptools import setup

DISTNAME = 'pypgmc'
DESCRIPTION = "Efficient Discrete Probabilistic Graphical Model inference for PyMC (2 and 3) . Offers Clique Tree and Loopy Belief Propagation Algorithms implemented via Theano"
LONG_DESCRIPTION    = """Efficient Discrete Probabilistic Graphical Model inference for PyMC (2 and 3).
            Offers efficient implementations of Clique Tree and Loopy Belief Propagation Algorithms
            implemented via Theano Expressions"""
MAINTAINER = 'Kai Londenberg'
MAINTAINER_EMAIL = 'Kai.Londenberg@googlemail.com'
AUTHOR = 'Kai Londenberg'
AUTHOR_EMAIL = 'Kai.Londenberg@googlemail.com'
URL = "http://github.com/pymc-devs/pymc"
LICENSE = "Apache 2.0 License "
VERSION = ""

classifiers = ['Development Status :: 1 - Alpha',
               'Programming Language :: Python',
               'Operating System :: OS Independent']

with open('requirements.txt') as f:
    required = f.read().splitlines()

if __name__ == "__main__":

    setup(name=DISTNAME,
          version=VERSION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          packages=['pypgmc'],
          classifiers=classifiers,
          install_requires=required)
