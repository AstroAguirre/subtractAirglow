# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:02:04 2020

@author: edcr4756
"""

#when promoting the widget in QtDesigner, it now depends on a file called mplwidget.py, which is why this is separate
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas #used to embed matplotlib figure into a Widget
from matplotlib.figure import Figure #used for same reason as above^

class MplWidget(QtWidgets.QWidget):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self,parent)
        self.canvas=FigureCanvas(Figure())
        vertical_layout=QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.canvas.axes=self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)