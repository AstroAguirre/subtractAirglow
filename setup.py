# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:39:44 2022

@author: edcr4756
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
      requirements=f.read().splitlines()

setup(
      name='subtractAirglow',
      version='0.0.2',
      description='Subtract airglow and reconstruct stellar emission from COS G130M spectra',
      url='https://github.com/AstroAguirre/subtractAirglow.git',
      author='Fernando Cruz Aguirre',
      author_email='edwin.cruzaguirre@lasp.colorado.edu',
      license='MIT',
      packages=find_packages(),
      package_data={'subtractAirglow' : ['*.dat','*.npy','*.ui','*.webp']},
      include_package_data=True,
      install_requires=requirements
      )
