# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:35:29 2021

@author: edcr4756
"""

#when promoting the widget in QtDesigner, it now depends on a file called mplwidget2.py, which is why this is separate
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas #used to embed matplotlib figure into a Widget
from matplotlib.figure import Figure #used for same reason as above^

class MplWidget2(QtWidgets.QWidget):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self,parent)
        self.canvas=FigureCanvas(Figure())
        vertical_layout=QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.canvas.axes1=self.canvas.figure.add_subplot(211)
        self.canvas.axes2=self.canvas.figure.add_subplot(212,sharex=self.canvas.axes1)
        self.setLayout(vertical_layout)