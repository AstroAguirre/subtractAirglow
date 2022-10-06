# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:39:44 2022

@author: edcr4756
"""

from setuptools import setup, find_packages

setup(
      name='subtractAirglow',
      version='0.0.1',
      description='Subtract airglow from COS G130M spectra',
      url='https://github.com/AstroAguirre/subtractAirglow.git',
      author='Fernando Cruz Aguirre',
      author_email='edwin.cruzaguirre@lasp.colorado.edu',
      license='MIT',
      packages=find_packages()
      )