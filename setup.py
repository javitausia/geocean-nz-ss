#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == '__main__':
    setup(name='sscode',
          version='1.0',
          description='Storm surge project code',
          author='Javier Tausia Hoyal',
          author_email='tausiaj@unican.es',
          url='https://github.com/javitausia/geocean-nz-ss'
    )