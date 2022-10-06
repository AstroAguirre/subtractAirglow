# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:02:04 2020

@author: edcr4756
"""

#when promoting the widget in QtDesigner, it now depends on a file called mplwidget.py, which is why this is separate
#https://www.youtube.com/watch?v=2C5VnE9wPhk
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas #used to embed matplotlib figure into a Widget
from matplotlib.figure import Figure #used for same reason as above^

class MplWidget(QtWidgets.QWidget):
    def __init__(self,parent=None): #not sure why I have to specify no parent, maybe it defaults to something
        QtWidgets.QWidget.__init__(self,parent)
        self.canvas=FigureCanvas(Figure())
        vertical_layout=QtWidgets.QVBoxLayout()
        #not sure if I need this if it's defined in the .ui, but the tutorial says so, so
        #actually, all I did in the .ui was say that a widget goes here so thats prolly it
        vertical_layout.addWidget(self.canvas)
        self.canvas.axes=self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)