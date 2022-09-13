#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='coadd',
      version='1.0.0',
      description='Coadd images',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/coadd',
      packages=['coadd'],
      package_dir={'':'python'},
      scripts=['bin/coadd'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','sep'],
      include_package_data=True
)
