# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:06:14 2021

@author: edcr4756
"""

#get path to files
import os
import subtractAirglow
path=os.path.dirname(subtractAirglow.__file__) #find path to package

import csv
import sys
import time
from subtractAirglow import voigt #import from the subtractAirglow package
import numpy as np
import pandas as pd
#import qdarktheme #https://pypi.org/project/pyqtdarktheme/
import matplotlib.pyplot as plt
from lmfit import Model
from scipy import integrate
from astropy.io import fits
from PyQt6 import QtWidgets, QtGui, QtCore
from astropy.modeling.models import Voigt1D, Gaussian1D
from recombinator.optimal_block_length import optimal_block_length
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar #used to generate the matplotlib toolbar

#import ui layouts
from subtractAirglow.AirglowRemovalUi import mainUi,bootstrapUi,missingUi,rangeUi,recoveredUi,residualsUi,resultsUi,stisUi

plt.ion() #make plt.draw() work in terminal/as package

class mainWindow(QtWidgets.QMainWindow,mainUi):
    def __init__(self):
        super(mainWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        
        #matplotlib toolbar
        self.addToolBar(NavToolbar(self.MplWidget.canvas,self))
        
        #File open push buttons
        self.openCOS.clicked.connect(self.fileCOS) #open COS x1d.fits or sav.txt file, G130M
        self.openSTIS.clicked.connect(self.fileSTIS) #open STIS x1d.fits file, E140M or G140M
        #get default text color so that GUI recoloring works in both light/dark mode
        textQColor=self.palette().text().color() #returns QColor with .red(),.green(),.blue() properties
        self.textColor='color: rgb(%d,%d,%d)' %(textQColor.red(),textQColor.green(),textQColor.blue())
        
        #lineEdits activate when pressing enter (maybe add an apply button?)
        self.starInput.textChanged.connect(self.starApply)
        self.starInput.setDisabled(True)
        
        #Which Line radio buttons, unavailable on startup
        self.radioLyA.toggled.connect(self.lineLyA)
        self.radioLyA.setDisabled(True)
        self.radioOI2.toggled.connect(self.lineOI2)
        self.radioOI2.setDisabled(True)
        self.radioOI5.toggled.connect(self.lineOI5)
        self.radioOI5.setDisabled(True)
        self.radioOI6.toggled.connect(self.lineOI6)
        self.radioOI6.setDisabled(True)
        
        #Range push button
        self.editRanges.clicked.connect(self.rangeOpen)
        self.editRanges.setDisabled(True)
        
        # #STIS radio buttons
        self.checkSTIS.clicked.connect(self.compareSTIS)
        self.checkSTIS.setDisabled(True)
        self.clearSTIS.clicked.connect(self.removeSTIS)
        self.clearSTIS.setDisabled(True)
        
        #Radial velocity lineEdit
        self.vradInput.returnPressed.connect(self.setVrad)
        self.vradInput.setDisabled(True)
        
        #ISM velocity lineEdit
        self.vismInput.returnPressed.connect(self.setVism)
        self.vismInput.setDisabled(True)
        
        #Fit mode radio buttons
        self.oneFit.toggled.connect(self.onePart)
        self.oneFit.setDisabled(True)
        self.twoFit.toggled.connect(self.twoPart)
        self.twoFit.setDisabled(True)
        
        #Airglow mode radio buttons
        self.autoAirmode.toggled.connect(self.autoMode)
        self.autoAirmode.setDisabled(True)
        self.manualAirmode.toggled.connect(self.manualMode)
        self.manualAirmode.setDisabled(True)     

        #Horizontal airglow slider stuff, unavailable on startup
        self.scaled_valH=round(self.airShift.value()*(1.5/1500.0),3) #slider goes from -1500..1500, scaled to -1.5..1.5, step size of 0.001
        self.shiftInput.setText(str(self.scaled_valH)) #display default value of 0.0
        self.shiftInput.returnPressed.connect(self.changeShift) #pressing enter on lineEdit box applies user shift
        self.shiftInput.setDisabled(True)
        self.airShift.sliderPressed.connect(self.shiftDis) #waits until the slider is pressed to disconnect
        self.airShift.sliderReleased.connect(self.shiftRec) #waits until the slider is released to reconnect
        self.airShift.setDisabled(True)
        
        #Vertical airglow slider stuff, unavailable on startup 
        self.scaled_valV=round(self.airScale.value()*(10.0/1000.0),2) #slider goes from 0..1000, scaled to 0.0..10.0, step size of 0.01
        self.scaleInput.setText(str(self.scaled_valV)) #display default value of 1.0
        self.scaleInput.returnPressed.connect(self.changeScale) #pressing enter on lineEdit box applies user scale
        self.scaleInput.setDisabled(True)
        self.airScale.sliderPressed.connect(self.scaleDis) #waits until the slider is pressed to disconnect
        self.airScale.sliderReleased.connect(self.scaleRec) #waits until the slider is released to reconnect
        self.airScale.setDisabled(True)
        
        #Fit shift slider stuff, unavailable on startup
        self.scaled_valFH=round(self.fitShift.value()*(1.5/1500),3) #slider goes from -1500..1500, scaled to -1.5..1.5, step size of 0.001
        self.fitshiftInput.setText(str(self.scaled_valFH)) #display default value of 0.0
        self.fitshiftInput.returnPressed.connect(self.applyShift) #pressing enter on lineEdit box applies user shift
        self.fitshiftInput.setDisabled(True)
        self.fitShift.valueChanged.connect(self.shiftFit) #waits for the value to change to shift
        self.fitShift.setDisabled(True)
        
        #Fit scale slider stuff, unavailable on startup
        self.scaled_valFV=round(self.fitScale.value()*(10.0/1000),2) #slider goes from 0..1000, scaled to 0.0..10.0, step size of 0.01
        self.fitscaleInput.setText(str(self.scaled_valFV)) #display default value of 1.0
        self.fitscaleInput.returnPressed.connect(self.applyScale) #pressing enter on lineEdit box applies user scale
        self.fitscaleInput.setDisabled(True)
        self.fitScale.valueChanged.connect(self.scaleFit) #waits for the value to change to scale
        self.fitScale.setDisabled(True)
        
        #Pre-Removal bush buttons, unavailable on startup
        self.fitComponents.clicked.connect(self.toggleComponents)
        self.fitComponents.setDisabled(True)
        self.fitResults.clicked.connect(self.showResults)
        self.fitResults.setDisabled(True)
        self.openCutoff.clicked.connect(self.readyCutoff)
        self.openCutoff.setDisabled(True)
        self.removeAirglow.clicked.connect(self.getTrue)
        self.removeAirglow.setDisabled(True)
        
        #Post-Removal push buttons, unavailable on startup
        self.removalPlots.clicked.connect(self.plotSubtraction)
        self.removalPlots.setDisabled(True)
        self.openBootstrap.clicked.connect(self.readyRRCBB)
        self.openBootstrap.setDisabled(True)
        self.saveData.clicked.connect(self.saveTrue)
        self.saveData.setDisabled(True)
        
        #Initialize booleans to prevent certain features
        self.readyCOS=False #prevent pretty much everything until file is loaded
        self.readySTIS=False #prevent anything involving STIS until file is loaded
        self.fitExists=False #prevent anthing involving changing the fit, i.e. sliders
        self.plotExists=False #prevents anything involving the plot before it exists
        self.useCutoff=[False,False,False,False] #checks to see if a cutoff value is applied
        self.bootDone=[False,False,False,False] #prevents saving an otherwise empty set of CSVs
        
        #Placeholders to be filled as the user runs the GUI
        self.whichLP=None #the LP of the observation
        self.whichSTIS=None #the STIS grating used for its observation
        self.waveCOS=None #the COS wavelength array
        self.fluxCOS=None #the COS flux array
        self.errrCOS=None #the COS flux error array
        self.waveSTIS=None #the STIS wavelength array
        self.fluxSTIS=None #the STIS flux array
        self.errrSTIS=None #the STIS flux error array
        self.currentLine=None #the current line to display on the plot for airglow removal
        self.lineLabel=None #the label of the line currently selected
        self.currentRange=None #the wavelengh range of the currently selected line
        self.radVelocity=None #the initial guess for the radial velocity of the star
        self.radISMVelocity=None #the initial guess for the ism velocity in the line of sight of the star
        self.whichFit=None #which fit is currently being run (one vs two part fitting)
        self.whichAir=None #which airglow mode is beung run (auto vs manual mode)
        self.lsfCOS=None #the COS LSF to be used for fitting
        self.lsfTrunc=None #the truncated COS LSF, if the fitting code determines truncation is needed
        self.stellarComp=None #the best fit pure stellar emission component
        self.selfreversalComp=None #the best fit self reversal absorption component
        self.ismComp=None #the best fit ISM absorption component
        self.airglowComp=None #the best fit airglow component
        self.airglowErrr=None #the best fit airglow component error
        self.convolvedComp=None #the convolved best fit minus the airglow
        self.fitIndex=None #this index selects which line shift/scale values to display
        
        #Booleans that check if certain conditions are true or false
        self.specialLyA=False #whether LyA is an LP3 1327 observation or not
        self.isM=False #whether a star is an M type or not, true=low SR, false=mid SR
        self.onlyA=False #whether COS data uses both sides of the detector or just side A (larger wavelengths)
        self.showComps=True #whether components are shown on the plot or not, defaulted to true
        self.fitOverride=False #when switching between lines, do not run a fit when changing slider values
        self.fitSlider=False #when recalcuating residuals/RRCBB after changing fit sliders, recalculate fit to the original lineMask
        self.bootRunning=False #when the bootstrap runs in 2 part mode, do not vary airglow params
        self.airglowRemoved=[False,False,False,False] #changes to true if the line has airglow removed
        
        #Initialize default values for variables used by the fit
        self.lineCenters=[1215.67,1302.1685,1304.8576,1306.0286] #wavelength centers of the spectral lines
        self.sfSTIS=[1.0,1.0,1.0,1.0] #the STIS scale factors to scale it to recovered COS data, default is 1.0
        self.allLinemasks=[[],[],[],[]] #the masks applied to each of the four lines
        self.allWaveinfs=[[],[],[],[]] #the "infinite resolution" wavegrids for each line
        self.allShifts=[0.0,0.0,0.0,0.0] #airglow shifts default to 0.0
        self.allScales=[1.0,1.0,1.0,1.0] #airglow scales default to 1.0 (may want to add another decimal to this scale to closer match auto mode)
        self.allFitshifts=[0.0,0.0,0.0,0.0] #fit shifts default to 0.0 
        self.allFitscales=[1.0,1.0,1.0,1.0] #fit scales default to 1.0
        self.allCutoffs=[np.inf,np.inf,np.inf,np.inf] #normalized residuals shouldn't be cut off at the start
        self.allCutmasks=[[],[],[],[]] #the masks produced from the normalized residuals, once created
        self.origMasks=[[],[],[],[]] #the original line masks before a cutoff is applied
        self.origBests=[[],[],[],[]] #the original best fit before a cutoff is applied
        self.origBerrs=[[],[],[],[]] #the original best fit error before a cutoff is applied
        
        #list of bools that will hold data going into the main CSV savefile
        self.finalLinemasks=[False,False,False,False] #line masks used (line + cutoff masks)
        self.finalWaveinfs=[False,False,False,False] #infinite resolution wavegrids
        self.finalStellar=[False,False,False,False] #best fit stellar components
        self.finalStelerr=[False,False,False,False] #best fit stellar component profile errors
        self.finalReversal=[False,False,False,False] #best fit self reversal components
        self.finalISM=[False,False,False,False] #best fit ism components
        self.finalAirglow=[False,False,False,False] #best fit airglow components
        self.finalAirgerr=[False,False,False,False] #best fit airglow errors
        self.finalParams=[False,False,False,False] #best fit parameters and errors
        self.finalBest=[False,False,False,False] #best fit profile
        self.finalBesterr=[False,False,False,False] #the errors of the best fit profile
        self.finalIfluxrecv=[False,False,False,False] #the integrated fluxes of the recovered profiles
        self.finalIfluxrerr=[False,False,False,False] #the errors of the recovered profile flux
        self.finalIfluxstel=[False,False,False,False] #the integrated fluxes of the best fit stellar+SR profiles
        self.finalIfluxserr=[False,False,False,False] #the errors of the stellar+SR integrated fluxes
        self.finalIfluxstis=[False,False,False,False] #the integrated fluxes of the integrated STIS data
        self.finalIfluxster=[False,False,False,False] #the errors of the integrated STIS flux
        self.finalSTISscale=[False,False,False,False] #the STIS scale factors used when plotting
        self.finalFitmode=[False,False,False,False] #keep track of which fit mode was used to get the parameters
        self.finalAirmode=[False,False,False,False] #keep track of which airglow mode was used to get the parameters
        self.finalEmethod=[False,False,False,False] #keep track of whether errors are LMFIT or RRCBB
        self.finalCutoffs=[False,False,False,False] #the normalized residual cutoffs used, if applicable
        self.finalUshifts=[False,False,False,False] #user inputted airglow shifts
        self.finalUscales=[False,False,False,False] #user inputted airglow scales
        self.finalUfitshifts=[False,False,False,False] #user inputted fit shifts
        self.finalUfitscales=[False,False,False,False] #user inputted fit scales
        self.finalRadial=[False,False,False,False] #user inputted stellar radial velocities
        self.finalRadism=[False,False,False,False] #user inputted ism radial velocities
        self.finalOptlen=[False,False,False,False] #RRCBB block length used
        self.finalNumblk=[False,False,False,False] #RRCBB number of blocks 
        self.finalNumpar=[False,False,False,False] #RRCBB Number of fit parameters
        self.finalNumsmp=[False,False,False,False] #RRCBB number of bootstrap samples created
        self.finalDeltaT=[False,False,False,False] #RRCBB time taken to complete all samples
        self.finalRCBfit=[False,False,False,False] #RRCBB best fits for all samples
        self.finalRCBste=[False,False,False,False] #RRCBB stellar components of best fits for all samples
        self.finalRCBpav=[False,False,False,False] #RRCBB best fit parameter values for all samples
        self.finalRCBpae=[False,False,False,False] #RRCBB best fit parameter errors
        self.finalRCBbfe=[False,False,False,False] #RRCBB error on the best fit profile
        self.finalRCBsce=[False,False,False,False] #RRCBB error on the stellar component profile
        self.finalRCBsie=[False,False,False,False] #RRCBB error of the integrated stellar flux
        
        #Initialize placeholders for plot features
        self.plotStarname='' #holds the name of the star
        self.rangeLyA=[1213.9,1217.4] #default LyA range
        self.rangeOI2=[1301.1,1303.5] #default OI2 range
        self.rangeOI5=[1303.5,1305.5] #default OI5 range    
        self.rangeOI6=[1305.5,1307.4] #default OI6 range
        self.currentXlim=(False,False) #xlim is saved as a tuple, use to keep current zoom/pan/etc.
        self.currentYlim=(False,False) #ylim is saved as a tuple, use to keep current zoom/pan/etc.
        
        self.show() #open the GUI window
        
    def cosReset(self):
        #reset all COS related control booleans
        self.readyCOS=False
        self.fitExists=False
        self.plotExists=False
        self.useCutoff=[False,False,False,False]
        self.bootDone=[False,False,False,False]
        
        #reset all COS related placeholders
        self.whichLP=None
        self.waveCOS=None
        self.fluxCOS=None
        self.errrCOS=None
        self.currentLine=None
        self.lineLabel=None
        self.currentRange=None
        self.radVelocity=None
        self.radISMVelocity=None
        self.whichFit=None
        self.whichAir=None
        self.lsfCOS=None
        self.lsfTrunc=None
        self.stellarComp=None
        self.selfreversalComp=None
        self.ismComp=None
        self.airglowComp=None
        self.airglowErrr=None
        self.convolvedComp=None
        self.fitIndex=None
        
        #reset all condition checks
        self.specialLyA=False
        self.isM=False
        self.onlyA=False
        self.showComps=True
        self.fitOverride=False
        self.fitSlider=False
        self.bootRunning=False
        self.airglowRemoved=[False,False,False,False]
        
        #reset default settings that could have changed
        self.rangeLyA=[1213.9,1217.4] 
        self.rangeOI2=[1301.1,1303.5] 
        self.rangeOI5=[1303.5,1305.5]     
        self.rangeOI6=[1305.5,1307.4] 
        self.allLinemasks=[[],[],[],[]]
        self.allWaveinfs=[[],[],[],[]]
        self.allShifts=[0.0,0.0,0.0,0.0]
        self.allScales=[1.0,1.0,1.0,1.0]
        self.allFitshifts=[0.0,0.0,0.0,0.0]
        self.allFitscales=[1.0,1.0,1.0,1.0]
        self.allCutoffs=[np.inf,np.inf,np.inf,np.inf]
        self.allCutmasks=[[],[],[],[]]
        self.origMasks=[[],[],[],[]]
        self.origBests=[[],[],[],[]]
        self.origBerrs=[[],[],[],[]]
        
        #reset the sliders and related values
        self.shiftInput.setText('0.0') #reset slider text box values
        self.scaleInput.setText('1.0')
        self.fitshiftInput.setText('0.0')
        self.fitscaleInput.setText('1.0')
        self.airShift.sliderPressed.disconnect() #disconnect to change without running fitting code
        self.airShift.sliderReleased.disconnect()
        self.airShift.setValue(int(round(0.0*(1500.0/1.5)))) #reset to default position
        self.airShift.sliderPressed.connect(self.shiftDis) 
        self.airShift.sliderReleased.connect(self.shiftRec) #reconnect when done
        self.airScale.sliderPressed.disconnect() 
        self.airScale.sliderReleased.disconnect()
        self.airScale.setValue(int(round(1.0*(1000.0/10.0))))
        self.airScale.sliderPressed.connect(self.scaleDis)
        self.airScale.sliderReleased.connect(self.scaleRec)
        self.fitShift.valueChanged.disconnect()
        self.fitShift.setValue(int(round(0.0*(1500.0/1.5))))
        self.fitShift.valueChanged.connect(self.shiftFit)
        self.fitScale.valueChanged.disconnect()
        self.fitScale.setValue(int(round(1.0*(1000.0/10.0))))
        self.fitScale.valueChanged.connect(self.scaleFit)
        self.scaled_valH=0.0 #reset the internal values
        self.scaled_valV=1.0
        self.scaled_valFH=0.0
        self.scaled_valFV=1.0
        
        #reset all final arrays for CSV saving
        self.finalLinemasks=[False,False,False,False]
        self.finalWaveinfs=[False,False,False,False]
        self.finalStellar=[False,False,False,False]
        self.finalStelerr=[False,False,False,False] 
        self.finalReversal=[False,False,False,False]
        self.finalISM=[False,False,False,False]
        self.finalAirglow=[False,False,False,False]
        self.finalAirgerr=[False,False,False,False]
        self.finalParams=[False,False,False,False]
        self.finalBest=[False,False,False,False]
        self.finalBesterr=[False,False,False,False]
        self.finalIfluxrecv=[False,False,False,False]
        self.finalIfluxrerr=[False,False,False,False]
        self.finalIfluxstel=[False,False,False,False]
        self.finalIfluxserr=[False,False,False,False]
        self.finalIfluxstis=[False,False,False,False] 
        self.finalIfluxster=[False,False,False,False]
        self.finalSTISscale=[False,False,False,False]
        self.finalFitmode=[False,False,False,False]
        self.finalAirmode=[False,False,False,False]
        self.finalEmethod=[False,False,False,False]
        self.finalCutoffs=[False,False,False,False]
        self.finalUshifts=[False,False,False,False]
        self.finalUscales=[False,False,False,False] 
        self.finalUfitshifts=[False,False,False,False] 
        self.finalUfitscales=[False,False,False,False] 
        self.finalRadial=[False,False,False,False] 
        self.finalRadism=[False,False,False,False] 
        self.finalOptlen=[False,False,False,False]
        self.finalNumblk=[False,False,False,False]
        self.finalNumpar=[False,False,False,False]
        self.finalNumsmp=[False,False,False,False]
        self.finalDeltaT=[False,False,False,False]
        self.finalRCBfit=[False,False,False,False]
        self.finalRCBste=[False,False,False,False]
        self.finalRCBpav=[False,False,False,False]
        self.finalRCBpae=[False,False,False,False]
        self.finalRCBbfe=[False,False,False,False]
        self.finalRCBsce=[False,False,False,False]
        self.finalRCBsie=[False,False,False,False]
        
        #reset all user inputted text fields
        self.vradInput.setText('')
        self.vismInput.setText('')
        self.starInput.setText('')
        
        #reset all edited labels
        self.lifeLabel.setText('Lifetime Position:')
        self.cen1327Label.setText('LP3 1327:')
        self.spectypeLabel.setText('M Type:')
        self.sideLabel.setText('Side A Only:')
        self.labelLyA.setText('      Lyα Range:')
        self.labelOI2.setText('OI 1302 Range:')
        self.labelOI5.setText('OI 1305 Range:')
        self.labelOI6.setText('OI 1306 Range:')
        
        #disable radio buttons not available on startup
        self.groupLine.setExclusive(False) #temporarily disable exclusivity to untoggle radio buttons
        self.radioLyA.setDisabled(True)
        if self.radioLyA.isChecked():
            self.radioLyA.toggle()
        self.radioOI2.setDisabled(True)
        if self.radioOI2.isChecked():
            self.radioOI2.toggle()
        self.radioOI5.setDisabled(True)
        if self.radioOI5.isChecked():
            self.radioOI5.toggle()
        self.radioOI6.setDisabled(True)
        if self.radioOI6.isChecked():
            self.radioOI6.toggle()
        self.groupLine.setExclusive(True)
        
        self.groupFit.setExclusive(False)
        self.oneFit.setDisabled(True)
        if self.oneFit.isChecked():
            self.oneFit.toggle()
        self.twoFit.setDisabled(True)
        if self.twoFit.isChecked():
            self.twoFit.toggle()
        self.groupFit.setExclusive(True)
        
        self.groupAirtype.setExclusive(False)
        self.autoAirmode.setDisabled(True)
        if self.autoAirmode.isChecked():
            self.autoAirmode.toggle()
        self.manualAirmode.setDisabled(True)
        if self.manualAirmode.isChecked():
            self.manualAirmode.toggle()
        self.groupAirtype.setExclusive(True)
        
        #disable buttons, lineEdits, and sliders not available on startup
        self.vradInput.setDisabled(True)
        self.vismInput.setDisabled(True)
        self.editRanges.setDisabled(True)
        self.shiftInput.setDisabled(True)
        self.airShift.setDisabled(True)
        self.scaleInput.setDisabled(True)
        self.airScale.setDisabled(True)
        self.fitshiftInput.setDisabled(True)
        self.fitShift.setDisabled(True)
        self.fitscaleInput.setDisabled(True)
        self.fitScale.setDisabled(True)
        self.fitComponents.setDisabled(True)
        self.fitResults.setDisabled(True)
        self.openCutoff.setDisabled(True)
        self.removeAirglow.setDisabled(True)
        self.removalPlots.setDisabled(True)
        self.openBootstrap.setDisabled(True)
        self.saveData.setDisabled(True)
        
        #remove emission line radio button tooltips, if any were applied
        self.radioLyA.setToolTip('')
        self.radioOI2.setToolTip('')
        self.radioOI5.setToolTip('')
        self.radioOI6.setToolTip('')
        
        #reset the main plot
        self.clearPlot()
        
    def stisReset(self):
        #reset STIS related booleans
        self.readySTIS=False
        
        #reset STIS related placeholders
        self.whichSTIS=None
        self.waveSTIS=None
        self.fluxSTIS=None
        self.errrSTIS=None
        
        #reset STIS related default values
        self.sfSTIS=[1.0,1.0,1.0,1.0]
        
        #reset all edited labels
        self.labelSTIS.setText('STIS E140M/G140M File:')
        self.modeSTIS.setText('STIS Mode:')
        
        #disable buttons not available on startup
        self.checkSTIS.setDisabled(True)
        self.clearSTIS.setDisabled(True)
        
        #remove G140M related tooltip, if applied
        self.checkSTIS.setToolTip('')
        
    def clearPlot(self):
        self.plotStarname=''
        self.plotExists=False
        self.currentXlim=(False,False)
        self.currentYlim=(False,False)
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.draw()
        
    def displayPlot(self,newXlim=False,newYlim=False):
        if isinstance(self.currentXlim[0],bool): #first time plotting
            self.MplWidget.canvas.axes.clear()
            if self.fitExists:
                self.MplWidget.canvas.axes.plot(self.waveCOS[self.normComp],self.fluxCOS[self.normComp],color='C0',linewidth=2,label='COS Data')
                self.MplWidget.canvas.axes.fill_between(self.waveCOS[self.normComp],(self.fluxCOS+self.errrCOS)[self.normComp],(self.fluxCOS-self.errrCOS)[self.normComp],color='C0',alpha=0.4)
                if self.whichFit==2:
                    self.MplWidget.canvas.axes.plot(self.waveCOS[self.normComp],self.oneSpectrum[self.normComp],color='C9',linewidth=2,linestyle='--',label='Recovered Spec.')
                    self.MplWidget.canvas.axes.fill_between(self.waveCOS[self.normComp],(self.oneSpectrum+self.oneError)[self.normComp],(self.oneSpectrum-self.oneError)[self.normComp],color='C9',alpha=0.4)                    
                self.MplWidget.canvas.axes.plot(self.waveModel+self.allFitshifts[self.fitIndex],self.bestFit*self.allFitscales[self.fitIndex],color='C1',linewidth=2,label='Best Fit')
                try:
                    self.MplWidget.canvas.axes.fill_between(self.waveModel+self.allFitshifts[self.fitIndex],(self.bestFit+self.intBesterr[0])*self.allFitscales[self.fitIndex],(self.bestFit-self.intBesterr[1])*self.allFitscales[self.fitIndex],color='C1',alpha=0.25)
                except:
                    pass
                self.MplWidget.canvas.axes.set_title(self.plotStarname+' G130M Data and Best Fit')
                if self.showComps:
                    self.MplWidget.canvas.axes.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.stellarComp*self.allFitscales[self.fitIndex],color='C5',linewidth=2,linestyle='-.',label='Stellar Emission')
                    if self.fitIndex<1:
                        self.MplWidget.canvas.axes.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.selfreversalComp*self.allFitscales[self.fitIndex],color='C6',linewidth=2,linestyle='-.',label='SR Attn. Emission')
                        self.MplWidget.canvas.axes.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.ismComp*self.allFitscales[self.fitIndex],color='C3',linewidth=2,linestyle='-.',label='SR+ISM Attn. Emission')
                    self.MplWidget.canvas.axes.plot(self.waveModel+self.allFitshifts[self.fitIndex],self.airglowComp*self.allFitscales[self.fitIndex],color='C4',linewidth=2,linestyle='--',label='Airglow Template')
            else:
                self.MplWidget.canvas.axes.plot(self.waveCOS,self.fluxCOS,color='C0',linewidth=2,label='COS Data')
                self.MplWidget.canvas.axes.fill_between(self.waveCOS,self.fluxCOS+self.errrCOS,self.fluxCOS-self.errrCOS,color='C0',alpha=0.4)
                self.MplWidget.canvas.axes.set_title(self.plotStarname+' G130M Data')
            self.MplWidget.canvas.axes.set_xlabel('Wavelength ($\AA$)')
            self.MplWidget.canvas.axes.set_ylabel('Flux Density (erg $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)')
            self.MplWidget.canvas.axes.legend()
            self.MplWidget.canvas.draw()
            
            self.currentXlim=self.MplWidget.canvas.axes.get_xlim()
            self.currentYlim=self.MplWidget.canvas.axes.get_ylim()
            
        else: #plot currently exists, so an xlim and ylim also exist
            if isinstance(newXlim,bool) and isinstance(newYlim,bool):
                #xlim and ylim not set by currentLine
                self.currentXlim=self.MplWidget.canvas.axes.get_xlim()
                self.currentYlim=self.MplWidget.canvas.axes.get_ylim()
            elif isinstance(newYlim,bool):
                #xlim is set by currentLine
                self.currentXlim=newXlim
                self.currentYlim=self.MplWidget.canvas.axes.get_ylim()
            elif isinstance(newXlim,bool):
                #ylim is set by currentLine
                self.currentXlim=self.MplWidget.canvas.axes.get_xlim()
                self.currentYlim=newYlim
            else:
                #xlim and ylim are set by currentLine
                self.currentXlim=newXlim
                self.currentYlim=newYlim
            
            self.MplWidget.canvas.axes.clear()
            if self.fitExists:
                self.MplWidget.canvas.axes.plot(self.waveCOS[self.normComp],self.fluxCOS[self.normComp],color='C0',linewidth=2,label='COS Data')
                self.MplWidget.canvas.axes.fill_between(self.waveCOS[self.normComp],(self.fluxCOS+self.errrCOS)[self.normComp],(self.fluxCOS-self.errrCOS)[self.normComp],color='C0',alpha=0.4)
                if self.whichFit==2:
                    self.MplWidget.canvas.axes.plot(self.waveCOS[self.normComp],self.oneSpectrum[self.normComp],color='C9',linewidth=2,linestyle='--',label='Recovered Spec.')
                    self.MplWidget.canvas.axes.fill_between(self.waveCOS[self.normComp],(self.oneSpectrum+self.oneError)[self.normComp],(self.oneSpectrum-self.oneError)[self.normComp],color='C9',alpha=0.4)                    
                self.MplWidget.canvas.axes.plot(self.waveModel+self.allFitshifts[self.fitIndex],self.bestFit*self.allFitscales[self.fitIndex],color='C1',linewidth=2,label='Best Fit')
                try:
                    self.MplWidget.canvas.axes.fill_between(self.waveModel+self.allFitshifts[self.fitIndex],(self.bestFit+self.intBesterr[0])*self.allFitscales[self.fitIndex],(self.bestFit-self.intBesterr[1])*self.allFitscales[self.fitIndex],color='C1',alpha=0.25)
                except:
                    pass
                self.MplWidget.canvas.axes.set_title(self.plotStarname+' G130M Data and Best Fit')
                if self.showComps:
                    self.MplWidget.canvas.axes.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.stellarComp*self.allFitscales[self.fitIndex],color='C5',linewidth=2,linestyle='-.',label='Stellar Emission')
                    if self.fitIndex<1:
                        self.MplWidget.canvas.axes.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.selfreversalComp*self.allFitscales[self.fitIndex],color='C6',linewidth=2,linestyle='-.',label='SR Attn. Emission')
                        self.MplWidget.canvas.axes.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.ismComp*self.allFitscales[self.fitIndex],color='C3',linewidth=2,linestyle='-.',label='SR+ISM Attn. Emission')
                    self.MplWidget.canvas.axes.plot(self.waveModel+self.allFitshifts[self.fitIndex],self.airglowComp*self.allFitscales[self.fitIndex],color='C4',linewidth=2,linestyle='--',label='Airglow Template')
            else:
                self.MplWidget.canvas.axes.plot(self.waveCOS,self.fluxCOS,color='C0',linewidth=2,label='COS Data')
                self.MplWidget.canvas.axes.fill_between(self.waveCOS,self.fluxCOS+self.errrCOS,self.fluxCOS-self.errrCOS,color='C0',alpha=0.4)
                self.MplWidget.canvas.axes.set_title(self.plotStarname+' G130M Data')
            self.MplWidget.canvas.axes.set_xlabel('Wavelength ($\AA$)')
            self.MplWidget.canvas.axes.set_ylabel('Flux Density (erg $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)')
            self.MplWidget.canvas.axes.set_xlim(self.currentXlim)
            self.MplWidget.canvas.axes.set_ylim(self.currentYlim)
            self.MplWidget.canvas.axes.legend()
            self.MplWidget.canvas.draw()
        
    def fileCOS(self): #if someone clicks X on this open window, make sure it doesnt screw everything over (meaning all the resetting should occur after a valid file has been opened)
        nameCOS=QtWidgets.QFileDialog.getOpenFileName(self,'Open COS G130M File','','X1D(*_x1d.fits);;Coadd(*sav.txt);;Fits(*fits);;Dat(*dat);;All(*)')
        if nameCOS[0][-9:]=='_x1d.fits' or nameCOS[0][-5:]=='.fits': #open x1d file, supports renaming these files
            
            self.cosReset()
            hdu=fits.open(nameCOS[0])
            hdr=hdu[0].header
            
            try:
                cos_grat=hdr['OPT_ELEM']
            except:
                cos_grat=False
                gratChoice=QtWidgets.QMessageBox.question(self,'Grating Not Found','The grating could not be found.\nContinue under the assumption that this is G130M data?',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
                if gratChoice==QtWidgets.QMessageBox.StandardButton.Yes:
                    cos_grat='G130M'
                else:
                    pass
                
            try:
                cos_name=hdr['TARGNAME']
            except:
                cos_name=False
                
            if cos_grat=='G130M':
                dat=hdu[1].data
                sideA=dat[0]
                if len(dat)==2: #assumes both sides    
                    sideB=dat[1]
                    self.sideLabel.setText('Side A Only: False')
                    #wavelength order is side B, side A
                    self.waveCOS=np.concatenate((sideB[3],sideA[3]),axis=0) 
                    self.fluxCOS=np.concatenate((sideB[4],sideA[4]),axis=0)
                    self.errrCOS=np.concatenate((sideB[5],sideA[5]),axis=0)
                    self.fileLP=hdu[0].header['LIFE_ADJ'] #integer
                    self.lifeLabel.setText('Lifetime Position: '+str(self.fileLP))
                    cenwave=hdu[0].header['CENWAVE']
                    if self.fileLP==3 and cenwave==1327:
                        self.specialLyA=True
                        self.cen1327Label.setText('LP3 1327: True')
                    else:
                        self.cen1327Label.setText('LP3 1327: False')
                elif len(dat)==1: #assumed side A only
                    self.onlyA=True
                    self.sideLabel.setText('Side A Only: True')
                    self.waveCOS=sideA[3]
                    self.fluxCOS=sideA[4]
                    self.errrCOS=sideA[5]
                    self.fileLP=hdu[0].header['LIFE_ADJ'] #integer
                    self.lifeLabel.setText('Lifetime Position: '+str(self.fileLP))
                    cenwave=hdu[0].header['CENWAVE']
                    if self.fileLP==3 and cenwave==1327:
                        self.specialLyA=True
                        self.cen1327Label.setText('LP3 1327: True')
                    else:
                        self.cen1327Label.setText('LP3 1327: False')
                else: #this case is for muscles data, for now
                    self.waveCOS=dat['WAVELENGTH']
                    self.fluxCOS=dat['FLUX']
                    self.errrCOS=dat['ERROR']
                    self.fileLP=False

                filename=nameCOS[0].split('/')[-1] #get name of file within folder
                self.filenameCOS=str(filename)
                self.labelCOS.setText('COS G130M File: '+self.filenameCOS)
                self.labelCOS.setStyleSheet(self.textColor)
                
                #if the target name can be grabbed from the file, put it on the lineEdit and apply
                if not isinstance(cos_name,bool):
                    self.starInput.setText(cos_name)
                    self.plotStarname=cos_name
                
                #if the LP cannot be grabbed, open dialog box
                if isinstance(self.fileLP,bool): 
                    #only checks for LP, but assumes that LP, starname, and all special conditions need to be checked
                    dialogFill=missingWindow()
                    dialogFill.exec()
                else:
                    #Only ask for M type, everything else is already determined
                    redDwarf=QtWidgets.QMessageBox.question(self,'Missing Information','Is this an M type star?',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
                    if redDwarf==QtWidgets.QMessageBox.StandardButton.Yes:
                        self.isM=True
                        self.spectypeLabel.setText('M Type: True')
                    else:
                        self.spectypeLabel.setText('M Type: False')
               
                #enable the spectral line radio buttons
                if not self.onlyA:
                    self.radioLyA.setDisabled(False)
                    self.vismInput.setDisabled(True)
                else:
                    self.radioLyA.setToolTip('Side A does not contain Lyα')
                self.radioOI2.setDisabled(False)
                self.radioOI5.setDisabled(False)
                self.radioOI6.setDisabled(False)
                
                #enable and display the line ranges
                self.editRanges.setDisabled(False)
                self.labelLyA.setText('      Lyα Range: '+str(self.rangeLyA[0])+' - '+str(self.rangeLyA[1])+' Å')
                self.labelOI2.setText('OI 1302 Range: '+str(self.rangeOI2[0])+' - '+str(self.rangeOI2[1])+' Å')
                self.labelOI5.setText('OI 1305 Range: '+str(self.rangeOI5[0])+' - '+str(self.rangeOI5[1])+' Å')
                self.labelOI6.setText('OI 1306 Range: '+str(self.rangeOI6[0])+' - '+str(self.rangeOI6[1])+' Å')
                
            
                #set internal GUI values    
                self.currenetRange=[min(self.waveCOS),max(self.waveCOS)]
                self.whichLP=self.fileLP
                self.readyCOS=True
                self.plotExists=True
                self.starInput.setDisabled(False)
                self.displayPlot()
                
                
            elif cos_grat==False:
                self.labelCOS.setText('COS G130M File: Could Not Read FITS File')
                self.labelCOS.setStyleSheet('color: red')
                self.readyCOS=False
                    
            else:
                self.labelCOS.setText('COS G130M File: Incorrect Grating')
                self.labelCOS.setStyleSheet('color: red')
                self.readyCOS=False
            
        elif nameCOS[0][-8:]=='.sav.txt': #opens IDL coadd file
        
            try:
                COSdata=pd.read_csv(nameCOS[0],skiprows=1,delimiter='\s+',names=['WAVE','FLUX','ERRR'])
                idlFile=True
            except:
                idlFile=False
            
            if idlFile:
                self.cosReset()
                COSdata=pd.read_csv(nameCOS[0],skiprows=1,delimiter='\s+',names=['WAVE','FLUX','ERRR'])
                self.waveCOS=np.array(COSdata['WAVE'])
                self.fluxCOS=np.array(COSdata['FLUX'])
                self.errrCOS=np.array(COSdata['ERRR'])
                
                self.fileLP=False
                dialogFill=missingWindow()
                dialogFill.exec()
                    
                #enable the line buttons
                if not self.onlyA:
                    self.radioLyA.setDisabled(False)
                else:
                    self.radioLyA.setToolTip('Side A does not contain Lyα')
                self.radioOI2.setDisabled(False)
                self.radioOI5.setDisabled(False)
                self.radioOI6.setDisabled(False)
                
                #enable and display the line ranges
                self.editRanges.setDisabled(False)
                self.labelLyA.setText('      Lyα Range: '+str(self.rangeLyA[0])+' - '+str(self.rangeLyA[1])+' Å')
                self.labelOI2.setText('OI 1302 Range: '+str(self.rangeOI2[0])+' - '+str(self.rangeOI2[1])+' Å')
                self.labelOI5.setText('OI 1305 Range: '+str(self.rangeOI5[0])+' - '+str(self.rangeOI5[1])+' Å')
                self.labelOI6.setText('OI 1306 Range: '+str(self.rangeOI6[0])+' - '+str(self.rangeOI6[1])+' Å')
                
                self.currenetRange=[min(self.waveCOS),max(self.waveCOS)]
                filename=nameCOS[0].split('/')[-1]
                self.filenameCOS=str(filename)
                self.labelCOS.setText('COS G130M File: '+self.filenameCOS)
                self.labelCOS.setStyleSheet(self.textColor)
                self.whichLP=self.fileLP
                self.readyCOS=True
                self.plotExists=True
                self.starInput.setDisabled(False)
                self.displayPlot()
            
        elif nameCOS[0][-4:]=='.dat': #opens .dat files with wavelength, flux, and error arrays
        
            try:
                COSdata=pd.read_csv(nameCOS[0],names=['WAV','FLX','ERR'],sep='\s+')
                datFile=True
            except:
                datFile=False
                
            if datFile:
                self.cosReset()
                COSdata=pd.read_csv(nameCOS[0],names=['WAV','FLX','ERR'],sep='\s+')
                self.waveCOS=np.array(COSdata['WAV'])
                self.fluxCOS=np.array(COSdata['FLX'])
                self.errrCOS=np.array(COSdata['ERR'])
                
                self.fileLP=False
                dialogFill=missingWindow()
                dialogFill.exec()
                    
                #enable the line buttons
                if not self.onlyA:
                    self.radioLyA.setDisabled(False)
                else:
                    self.radioLyA.setToolTip('Side A does not contain Lyα')
                self.radioOI2.setDisabled(False)
                self.radioOI5.setDisabled(False)
                self.radioOI6.setDisabled(False)
                
                #enable and display the line ranges
                self.editRanges.setDisabled(False)
                self.labelLyA.setText('      Lyα Range: '+str(self.rangeLyA[0])+' - '+str(self.rangeLyA[1])+' Å')
                self.labelOI2.setText('OI 1302 Range: '+str(self.rangeOI2[0])+' - '+str(self.rangeOI2[1])+' Å')
                self.labelOI5.setText('OI 1305 Range: '+str(self.rangeOI5[0])+' - '+str(self.rangeOI5[1])+' Å')
                self.labelOI6.setText('OI 1306 Range: '+str(self.rangeOI6[0])+' - '+str(self.rangeOI6[1])+' Å')
                
                self.currenetRange=[min(self.waveCOS),max(self.waveCOS)]
                filename=nameCOS[0].split('/')[-1]
                self.filenameCOS=str(filename)
                self.labelCOS.setText('COS G130M File: '+self.filenameCOS)
                self.labelCOS.setStyleSheet(self.textColor)
                self.whichLP=self.fileLP
                self.readyCOS=True
                self.plotExists=True
                self.starInput.setDisabled(False)
                self.displayPlot()
        
        else:
            self.labelCOS.setText('COS G130M File: Incorrect or Unsupported File Format')
            self.labelCOS.setStyleSheet('color: red')
            self.readyCOS=False
            self.clearPlot()
            
        if self.readyCOS: #check if the OI lines can be selected 
            lmOI2=((self.rangeOI2[0]<=self.waveCOS)&(self.waveCOS<=self.rangeOI2[1])) #line mask
            fwOI2=1.0/self.errrCOS[lmOI2] #fit weights
            if np.any(fwOI2==np.inf):
                self.radioOI2.setDisabled(True)
                self.radioOI2.setToolTip('The current wavelength range for this line contains flux error values that are zero, try editing the line range')
            lmOI5=((self.rangeOI5[0]<=self.waveCOS)&(self.waveCOS<=self.rangeOI5[1])) #line mask
            fwOI5=1.0/self.errrCOS[lmOI5] #fit weights
            if np.any(fwOI5==np.inf):
                self.radioOI5.setDisabled(True)
                self.radioOI5.setToolTip('The current wavelength range for this line contains flux error values that are zero, try editing the line range')
            lmOI6=((self.rangeOI6[0]<=self.waveCOS)&(self.waveCOS<=self.rangeOI6[1])) #line mask
            fwOI6=1.0/self.errrCOS[lmOI6] #fit weights
            if np.any(fwOI6==np.inf):
                self.radioOI6.setDisabled(True)
                self.radioOI6.setToolTip('The current wavelength range for this line contains flux error values that are zero, try editing the line range')
        
    def fileSTIS(self):
        nameSTIS=QtWidgets.QFileDialog.getOpenFileName(self,'Open STIS E140M/G140M File','','X1D(*_x1d.fits);;Fits(*fits);;Dat(*dat);;All(*)')
        if nameSTIS[0][-9:]=='_x1d.fits' or nameSTIS[0][-5:]=='.fits': #open x1d file, supports renaming these files
            
            self.stisReset()
        
            hdu=fits.open(nameSTIS[0])
            hdr=hdu[0].header
            data=hdu[1].data
            wave=data['wavelength'].flatten()
            flux=data['flux'].flatten()
            errr=data['error'].flatten()  
            sort_key=np.argsort(wave)
            self.waveSTIS=wave[sort_key]
            self.fluxSTIS=flux[sort_key]
            self.errrSTIS=errr[sort_key] 
            
            try:
                stis_grt=hdr['OPT_ELEM'] #standard STIS format
                stis_muscles=False
            except:
                try:
                    stis_grt=hdr['GRATING'] #muscles format
                    stis_muscles=True #used if this is muscles data
                except:
                    whichGrat=QtWidgets.QMessageBox()
                    whichGrat.setIcon(QtWidgets.QMessageBox.Icon.Question)
                    whichGrat.setWindowTitle('Which STIS Grating was Used?')
                    whichGrat.setText('Unable to determine which STIS grating\nwas used, select a grating below.')
                    whichGrat.addButton('G140M',QtWidgets.QMessageBox.ButtonRole.AcceptRole) #give G140M accept
                    whichGrat.addButton('E140M',QtWidgets.QMessageBox.ButtonRole.RejectRole) #give E140M reject
                    stisGrat=whichGrat.exec()
                    if stisGrat==0: #returned value for accept
                        stis_grt='G140M'
                    elif stisGrat==1: #returned value for reject
                        stis_grt='E140M'
                    stis_muscles=False

            if stis_grt=='G140M' and not stis_muscles:
                filename2=nameSTIS[0].split('/')[-1]
                self.filenameSTIS=str(filename2)
                self.labelSTIS.setText('STIS E140M/G140M File: '+self.filenameSTIS)
                self.whichSTIS='G'
                self.modeSTIS.setText('STIS Mode: G140M')
                self.sfSTIS=[1.0,1.0,1.0,1.0]
                self.readySTIS=True
                if self.currentLine!=None: #only disable if COS data is loaded and a line has been selected
                    if self.whichSTIS=='G' and self.currentLine>=2: #for G140M data, do not enable if OI lines are selected
                        self.checkSTIS.setToolTip('G140M data does not contain OI emission lines')
                    else: #otherwise, enable the button
                        self.checkSTIS.setDisabled(False)
                self.clearSTIS.setDisabled(False)
                
            elif stis_muscles:
                filename2=nameSTIS[0].split('/')[-1]
                self.filenameSTIS=str(filename2)
                if stis_grt=='G140M':
                    self.labelSTIS.setText('STIS E140M/G140M File: '+self.filenameSTIS)
                    self.whichSTIS='G'
                    self.modeSTIS.setText('STIS Mode: G140M')
                    self.sfSTIS=[1.0,1.0,1.0,1.0]
                    self.readySTIS=True
                    if self.readyCOS:
                        self.checkSTIS.setDisabled(False)
                    self.clearSTIS.setDisabled(False)
                elif stis_grt=='E140M':
                    self.labelSTIS.setText('STIS E140M/G140M File: '+self.filenameSTIS)
                    self.whichSTIS='E'
                    self.modeSTIS.setText('STIS Mode: E140M')
                    self.sfSTIS=[1.0,1.0,1.0,1.0]
                    self.readySTIS=True
                    if self.readyCOS:
                        self.checkSTIS.setDisabled(False)
                    self.clearSTIS.setDisabled(False)
                else:
                    self.labelSTIS.setText('STIS E140M/G140M File: Incorrect Grating')
                    self.labelSTIS.setStyleSheet('color: red')
                    self.modeSTIS.setText('STIS Mode:')
                    self.checkSTIS.setDisabled(True)
                    self.whichSTIS=None
                    self.sfSTIS=[1.0,1.0,1.0,1.0]
                    self.readySTIS=False
                
                
            elif stis_grt=='E140M' and not stis_muscles:
                filename2=nameSTIS[0].split('/')[-1]
                self.filenameSTIS=str(filename2)
                self.labelSTIS.setText('STIS E140M/G140M File: '+self.filenameSTIS)
                self.whichSTIS='E'
                self.modeSTIS.setText('STIS Mode: E140M')
                self.sfSTIS=[1.0,1.0,1.0,1.0]
                self.readySTIS=True
                if self.readyCOS:
                    self.checkSTIS.setDisabled(False)
                self.clearSTIS.setDisabled(False)
            
            else:
                self.labelSTIS.setText('STIS E140M/G140M File: Incorrect Grating')
                self.labelSTIS.setStyleSheet('color: red')
                self.modeSTIS.setText('STIS Mode:')
                self.checkSTIS.setDisabled(True)
                self.whichSTIS=None
                self.sfSTIS=[1.0,1.0,1.0,1.0]
                self.readySTIS=False
        
        elif nameSTIS[0][-4:]=='.dat': #opens .dat files with wavelength, flux, and error arrays
        
            self.stisReset()
            
            STISdata=pd.read_csv(nameSTIS[0],names=['WAV','FLX','ERR'],sep='\s+')
            self.waveSTIS=np.array(STISdata['WAV'])
            self.fluxSTIS=np.array(STISdata['FLX'])
            self.errrSTIS=np.array(STISdata['ERR'])
            
            #data files will not contain additional info by design, need to ask which grating was used
            whichGrat=QtWidgets.QMessageBox()
            whichGrat.setIcon(QtWidgets.QMessageBox.Icon.Question)
            whichGrat.setWindowTitle('Which STIS Grating was Used?')
            whichGrat.setText('Unable to determine which STIS grating\nwas used, select a grating below.')
            whichGrat.addButton('G140M',QtWidgets.QMessageBox.ButtonRole.AcceptRole) #give G140M accept
            whichGrat.addButton('E140M',QtWidgets.QMessageBox.ButtonRole.RejectRole) #give E140M reject
            stisGrat=whichGrat.exec()
            if stisGrat==0: #returned value for accept
                stis_grt='G140M'
            elif stisGrat==1: #returned value for reject
                stis_grt='E140M'
            
            if stis_grt=='G140M':
                filename2=nameSTIS[0].split('/')[-1]
                self.filenameSTIS=str(filename2)
                self.labelSTIS.setText('STIS E140M/G140M File: '+self.filenameSTIS)
                self.whichSTIS='G'
                self.modeSTIS.setText('STIS Mode: G140M')
                self.sfSTIS=[1.0,1.0,1.0,1.0]
                self.readySTIS=True
                if self.currentLine!=None: #only disable if COS data is loaded and a line has been selected
                    if self.whichSTIS=='G' and self.currentLine>=2: #for G140M data, do not enable if OI lines are selected
                        self.checkSTIS.setToolTip('G140M data does not contain OI emission lines')
                    else: #otherwise, enable the button
                        self.checkSTIS.setDisabled(False)
                self.clearSTIS.setDisabled(False)
                
            elif stis_grt=='E140M':
                filename2=nameSTIS[0].split('/')[-1]
                self.filenameSTIS=str(filename2)
                self.labelSTIS.setText('STIS E140M/G140M File: '+self.filenameSTIS)
                self.whichSTIS='E'
                self.modeSTIS.setText('STIS Mode: E140M')
                self.sfSTIS=[1.0,1.0,1.0,1.0]
                self.readySTIS=True
                if self.readyCOS:
                    self.checkSTIS.setDisabled(False)
                self.clearSTIS.setDisabled(False)
        
        else:
            self.labelSTIS.setText('STIS E140M/G140M File: Incorrect File Format')
            self.labelSTIS.setStyleSheet('color: red')
            self.modeSTIS.setText('STIS Mode:')
            self.checkSTIS.setDisabled(True)
            self.whichSTIS=None
            self.sfSTIS=[1.0,1.0,1.0,1.0]
            self.readySTIS=False
            
        if self.fitExists and self.readySTIS: #if fit was done, then user uploads STIS, iFlux won't be calculated
            maskSTIS=((self.currentRange[0]<=self.waveSTIS)&(self.waveSTIS<=self.currentRange[1]))
            try: #if no STIS data is present within the current range, cannot integrate
                self.intSTISdat=self.integrateFlux(self.waveSTIS[maskSTIS],self.fluxSTIS[maskSTIS]) #sfSTIS will be set to 1.0 when successfully reading in data, do not need to scale here
                self.intSTISerr=self.coreEnforcer(self.waveSTIS[maskSTIS],self.fluxSTIS[maskSTIS],self.errrSTIS[maskSTIS])
            except:
                self.intSTISdat=0.0
                self.intSTISerr=[False,False]
    
    def starApply(self):
        self.plotStarname=self.starInput.text()
        if self.plotExists:
            self.displayPlot()
                
    def selectLSF(self,whichOut=0):
        if self.specialLyA and self.fitIndex==0:
            lsf_fname=path+'//aa_LSFTable_G130M_1327_LP3_cn.dat' #only for LP3 1327 specifically for LyA
            lsf_waves=['1217','1302','1307'] 
        else:
            lsf_fname=path+'//aa_LSFTable_G130M_1291_LP'+str(self.whichLP)+'_cn.dat' #otherwise, always use the LPx 1291 LSF for LyA and OI
            if self.whichLP==5:
                lsf_waves=['1215','1300','1305'] 
            else:
                lsf_waves=['1214','1300','1305'] 
            
        lsfOpen=pd.read_csv(lsf_fname,sep=' ')
        if self.fitIndex==0:        
            lsfCOS_f=np.array(lsfOpen[lsf_waves[0]])
        elif self.fitIndex==1:
            lsfCOS_f=np.array(lsfOpen[lsf_waves[1]])
        elif self.fitIndex==2 or self.fitIndex==3:
            lsfCOS_f=np.array(lsfOpen[lsf_waves[2]])
            
        if whichOut!=0:
            lsfCOS_f=float(lsf_waves[0]) #only get the central wavelength as the output
        
        return lsfCOS_f    

    def wave_cos_lsf(self,lsf_arr,lsf_cen):
        #G130M has a 9.97mA/px dispersion, StSCI provided ones assume this dispersion
        a_px=0.00997
        indval=np.where(lsf_arr==np.max(lsf_arr))[0][0]
        r_half=a_px*np.arange(0,len(lsf_arr)-indval,1)
        l_half=a_px*np.arange(-indval,0,1)
        wlarr=np.append(l_half,r_half)
        return wlarr

    def ready_cos_lsf(self,orig_lsf_wave,orig_lsf,data_wave,data_wave_spacing=0.00997):
      data_wave_length = len(data_wave)
      lsf_lam_min = np.round(np.min(orig_lsf_wave)/data_wave_spacing) * data_wave_spacing
      lsf_lam_onesided = np.arange(lsf_lam_min,0,data_wave_spacing)  ### Make sure it's even and doesn't include zero
      if len(lsf_lam_onesided) % 2 != 0:
        lsf_lam_onesided = lsf_lam_onesided[1::] # get rid of the first element of the array
    
      lsf_lam_flipped = lsf_lam_onesided[::-1]
      lsf_lam_pluszero=np.append(lsf_lam_onesided,np.array([0]))
      lsf_lam=np.append(lsf_lam_pluszero,lsf_lam_flipped) # should be odd
    
      lsf_interp = np.interp(lsf_lam,orig_lsf_wave,orig_lsf/np.sum(orig_lsf))
      lsf_interp_norm = lsf_interp/np.sum(lsf_interp)
    
      if data_wave_length < len(lsf_interp_norm):
          lsf_interp_norm = np.delete(lsf_interp_norm,np.where(lsf_interp_norm == 0))
          lsf_interp_norm = np.insert(lsf_interp_norm,0,0)
          lsf_interp_norm = np.append(lsf_interp_norm,0)
    
      return lsf_interp_norm

    def prepareLSF(self):
        lsfval=self.selectLSF()
        lsfcen=self.selectLSF(whichOut=1)
        lsfwav=self.wave_cos_lsf(lsfval,lsfcen)
        self.waveInf=np.arange(self.currentRange[0],self.currentRange[1]+0.001,0.001)
        self.lsfCOS=self.ready_cos_lsf(lsfwav,lsfval,self.waveInf,0.001)
        self.allWaveinfs[self.fitIndex]=self.waveInf
        
    def fillFalse(self,arr,fill):
        #make the smaller fill array be the size of the arr array by filling the sides with enough falses
        leftlen=0
        ritelen=0
        swapside=False
        for i in arr:
            if not i and not swapside:
                leftlen+=1 #count the number of falses to the left
            elif i and not swapside:
                swapside=True #switch to the right
            elif not i and swapside:
                ritelen+=1 #count number of falses to the right
            else:
                pass
        
        leftarr=[False]*leftlen
        ritearr=[False]*ritelen
        filled=np.concatenate((leftarr,fill,ritearr)) #combine together to make fill the size of arr
        return filled
                
    def fitChanges(self):
        #if one of the 9 user input values changes, normalized residuals and RRCBB become invalud if already applied/ran
        #these 9 values are the airglow shift/scale, the fit shift/scale, the two RVs, the two fit modes, and the line range
        if self.useCutoff[self.fitIndex]==True and self.bootDone[self.fitIndex]==True:
            fitChange=QtWidgets.QMessageBox.question(self,'Change fit value?','Changing this value will reset the normalized residuals\nand will require a new RRCBB run.\nAre you sure you want to change this?',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
            if fitChange==QtWidgets.QMessageBox.StandardButton.Yes:
                self.resetResid(self.fitIndex)
                self.resetRRCBB(self.fitIndex)
                return True
            else:
                return False
        
        elif self.useCutoff[self.fitIndex]==True:
            fitChange=QtWidgets.QMessageBox.question(self,'Change fit value?','Changing this value will reset the normalized residuals.\nAre you sure you want to change this?',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
            if fitChange==QtWidgets.QMessageBox.StandardButton.Yes:
                self.resetResid(self.fitIndex)
                return True
            else:
                return False
            
        elif self.bootDone[self.fitIndex]==True:
            fitChange=QtWidgets.QMessageBox.question(self,'Change fit value?','Changing this value will require a new RRCBB run.\nAre you sure you want to change this?',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
            if fitChange==QtWidgets.QMessageBox.StandardButton.Yes:
                self.resetRRCBB(self.fitIndex)
                return True
            else:
                return False
            
        else:
            return True #if NRC/RRCBB haven't happened, proceed as normal

    def resetResid(self,lineInd):
        #reset relevant parameters
        self.useCutoff[lineInd]=False
        self.allCutoffs[lineInd]=np.inf
        self.allCutmasks[lineInd]=[]
        self.finalCutoffs[lineInd]=False
        self.origMasks[lineInd]=[]
        self.origBests[lineInd]=[]
        self.origBerrs[lineInd]=[]
        #reset line masks and LSF (can't call lineActions or else airglow sliders get a double disconnect, breaking the GUI)
        self.normComp=[True]*len(GUI.waveCOS)
        self.lineMask=(self.lineComp&self.normComp)
        self.allLinemasks[self.fitIndex]=self.lineMask
        self.prepareLSF()
        if self.fitSlider==True:
            #model needs to rerun after changing fit sliders
            #only needed for these sliders since everything else already reruns the model by design
            if self.whichFit==1:
                self.runModel()
            else:
                self.runModel2()
            self.displayPlot()
        
        
    def resetRRCBB(self,lineInd):
        #reset relevant parameters
        self.bootDone[lineInd]=False
        self.finalEmethod[lineInd]=False
        self.finalOptlen[lineInd]=False
        self.finalNumblk[lineInd]=False
        self.finalNumpar[lineInd]=False
        self.finalNumsmp[lineInd]=False
        self.finalDeltaT[lineInd]=False
        self.finalRCBfit[lineInd]=False
        self.finalRCBste[lineInd]=False
        self.finalRCBpav[lineInd]=False
        self.finalRCBpae[lineInd]=False
        self.finalRCBbfe[lineInd]=False
        self.finalRCBsce[lineInd]=False
        self.finalRCBsie[lineInd]=False
        
    
    def lineActions(self,lineRange,lineIndex):
        self.lineComp=((lineRange[0]<=self.waveCOS)&(self.waveCOS<=lineRange[1]))
        if self.useCutoff[lineIndex]:
            self.normComp=self.fillFalse(self.lineComp,self.allCutmasks[lineIndex])
        else:
            self.normComp=[True]*len(GUI.waveCOS) #does nothing
        self.lineMask=(self.lineComp&self.normComp)
        ytop=1.1*np.max(self.fluxCOS[self.lineMask])
        ybot=-2e-16
        self.currentRange=lineRange
        self.allLinemasks[self.fitIndex]=self.lineMask
        self.prepareLSF()
        
        if self.fitExists:
            #reset sliders to the saved shift and scale of each line
            self.airShift.setValue(int(round(self.allShifts[lineIndex]*(1500.0/1.5))))
            self.airScale.setValue(int(round(self.allScales[lineIndex]*(1000.0/10.0))))
            self.fitShift.setValue(int(round(self.allFitshifts[lineIndex]*(1500.0/1.5))))
            self.fitScale.setValue(int(round(self.allFitscales[lineIndex]*(1000.0/10.0))))
            self.fitOverride=True
            self.shiftRec() #update the sliders with the correct values for the selected line
            self.scaleRec()
            self.fitOverride=False #prevent fit from running 3 times when switching lines
            
            if self.whichFit==1:
                self.runModel()
            else:
                self.runModel2()
                
            self.displayPlot(self.currentRange,[ybot,ytop])
        else:
            self.displayPlot(self.currentRange,[ybot,ytop])
            
        if self.airglowRemoved[lineIndex]:
            self.removalPlots.setDisabled(False)
            self.openBootstrap.setDisabled(False)
            self.saveData.setDisabled(False)
        else: #only allow these buttons to be used if the selected line has had airglow removed
            self.removalPlots.setDisabled(True)
            self.openBootstrap.setDisabled(True)
            self.saveData.setDisabled(True)
        
    def lineLyA(self,selected):
        if selected:
            self.lineLabel='Lyα'
            self.fitIndex=0
            
            if self.currentLine==None:
                self.vradInput.setDisabled(False)
            else:
                self.vismInput.setText(str(self.radISMVelocity)) #if OI is selected and then LyA selected, need to show the default value in the box
            self.vismInput.setDisabled(False) #reenable RV ISM when switching to LyA for first time or when copming from OI line
            self.vismInput.setToolTip('')
            if self.readySTIS:
                self.checkSTIS.setDisabled(False)
                self.checkSTIS.setToolTip('') #no tooltip needed here, overwrites an OI related tooltip
            
            if self.specialLyA:
                self.currentLine=1 #special case of LyA
            else:
                self.currentLine=0 #typical case of LyA
                
            self.lineActions(self.rangeLyA,self.fitIndex)
                
                
    def lineOI2(self,selected):
        if selected:
            self.lineLabel='OI 1302'
            self.fitIndex=1
            if self.currentLine==None:
                self.vradInput.setDisabled(False)
                self.radISMVelocity=0.0 #set a default value, for fit to not be upset
            else:
                self.vismInput.setDisabled(True) #if enabled by LyA, disable it for OI
                if self.radISMVelocity==None:
                    self.radISMVelocity=0.0 #If LyA is selected but value isn't inputted, input one so nothing breaks
            self.vismInput.setToolTip('ISM Velocity is not required for OI emission lines')
            if self.readySTIS:
                if self.whichSTIS=='E':
                    self.checkSTIS.setDisabled(False)
                elif self.whichSTIS=='G':
                    self.checkSTIS.setDisabled(True)
                    self.checkSTIS.setToolTip('G140M data does not contain OI emission lines')
                        
            self.currentLine=2 
            self.lineActions(self.rangeOI2,self.fitIndex)
            
    def lineOI5(self,selected):
        if selected:
            self.lineLabel='OI 1305'
            self.fitIndex=2
            if self.currentLine==None:
                self.vradInput.setDisabled(False)
                self.radISMVelocity=0.0 #set a default value, for fit to not be upset
            else:
                self.vismInput.setDisabled(True) #if enabled by LyA, disable it for OI but keep LyA RV ISM value
            self.vismInput.setToolTip('ISM Velocity is not required for OI emission lines')
            if self.readySTIS:
                if self.whichSTIS=='E':
                    self.checkSTIS.setDisabled(False)
                elif self.whichSTIS=='G':
                    self.checkSTIS.setDisabled(True)
                    self.checkSTIS.setToolTip('G140M data does not contain OI emission lines')
            self.currentLine=5
            self.lineActions(self.rangeOI5,self.fitIndex)
            
    def lineOI6(self,selected):
        if selected:
            self.lineLabel='OI 1306'
            self.fitIndex=3
            if self.currentLine==None:
                self.vradInput.setDisabled(False)
                self.radISMVelocity=0.0 #set a default value, for fit to not be upset
            else:
                self.vismInput.setDisabled(True) #if enabled by LyA, disable it for OI but keep LyA RV ISM value
            self.vismInput.setToolTip('ISM Velocity is not required for OI emission lines')
            if self.readySTIS:
                if self.whichSTIS=='E':
                    self.checkSTIS.setDisabled(False)
                elif self.whichSTIS=='G':
                    self.checkSTIS.setDisabled(True)
                    self.checkSTIS.setToolTip('G140M data does not contain OI emission lines')
            self.currentLine=6 
            self.lineActions(self.rangeOI6,self.fitIndex)
            
    def rangeOpen(self):
        dialogRange=rangeWindow()
        dialogRange.exec()
        
    def compareSTIS(self):
        #becuase this is a main window, I need to link it to the parent main window (or else it will close randomly)
        self.dialogSTIS=stisWindow()
        self.dialogSTIS.show()
        
    def removeSTIS(self):
        deleteSTIS=QtWidgets.QMessageBox.question(self,'Clear STIS Data?','Are you sure you want to clear the current STIS data?',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
        if deleteSTIS==QtWidgets.QMessageBox.StandardButton.Yes:
            self.stisReset()
        else:
            pass
        
    def setVrad(self):
        try:
            radVel=float(self.vradInput.text())
            isFloat=True
        except:
            notFloat=QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                isFloat=False
                
        if isFloat:
            if self.whichFit==None and isinstance(self.radISMVelocity,float):
                #enable fit selection once both radial velocities are inputted and a line has been selected
                self.oneFit.setDisabled(False)
                self.twoFit.setDisabled(False)
                self.autoAirmode.setDisabled(False)
                self.manualAirmode.setDisabled(False)
            QtWidgets.QApplication.focusWidget().clearFocus()
    
            if self.fitExists:
                #user input, make sure change is okay
                if radVel!=self.radVelocity:
                    proceed=self.fitChanges()
                else:
                    proceed=False #if value is the same, then no need to recalculate the same fit
                if proceed:
                    self.radVelocity=float(self.vradInput.text())
                    
                    if self.whichFit==1:
                        self.runModel()
                    else:
                        self.runModel2()
                    self.displayPlot()
                else:
                    self.vradInput.setText(str(self.radVelocity)) #return original value back to the input box
            else:
                #before fit exists, this check doesn't matter, so this can be set without issue
                self.radVelocity=float(self.vradInput.text())
                
    def setVism(self):
        try:
            radISM=float(self.vismInput.text())
            isFloat=True
        except:
            notFloat=QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                isFloat=False
                
        if isFloat:
            if self.whichFit==None and isinstance(self.radVelocity,float):
                #enable fit selection once both radial velocities are inputted and a line has been selected
                self.oneFit.setDisabled(False)
                self.twoFit.setDisabled(False)
                self.autoAirmode.setDisabled(False)
                self.manualAirmode.setDisabled(False)
            QtWidgets.QApplication.focusWidget().clearFocus()
    
            if self.fitExists:
                #user input, make sure change is okay
                if radISM!=self.radISMVelocity:
                    proceed=self.fitChanges()
                else:
                    proceed=False #if value is the same, then no need to recalculate the same fit
                if proceed:
                    self.radISMVelocity=float(self.vismInput.text())
                    
                    if self.whichFit==1:
                        self.runModel()
                    else:
                        self.runModel2()
                    self.displayPlot()
                else:
                    self.vismInput.setText(str(self.radISMVelocity)) #return original value back to the input box
            else:
                self.radISMVelocity=float(self.vismInput.text())
                
    def stellarComponent(self,waveGrid,lineCen,radialVel,fwhmG,fwhmL,fluxAmp):
        lineCen=lineCen*((radialVel/3e5)+1) #redshifted line center
        if self.fitIndex==0: #for LyA
            fwhmGA=(lineCen/3e5)*fwhmG #Gaussian FWHM, in AA
            fwhmLA=(lineCen/3e5)*fwhmL #Lorentzian FWHM, in AA
            starProfile=Voigt1D(x_0=lineCen,amplitude_L=10**fluxAmp,fwhm_L=fwhmLA,fwhm_G=fwhmGA)
        else: #for OI Triplet
            stdvGA=(lineCen/3e5)*(fwhmG/2.3548)
            starProfile=Gaussian1D(amplitude=10**fluxAmp,mean=lineCen,stddev=stdvGA)
            
        stellarProfile=starProfile(waveGrid)
    
        return stellarProfile 
    
    def tauProfile(self,colDens,velShifts,dopbParam,whichLine):
        if whichLine=='h1':
            lam0s,fs,gammas=1215.67,0.4161,6.26e8
        elif whichLine=='d1':
            lam0s,fs,gammas=1215.3394,0.4161,6.27e8
            
        totN=10.0**colDens
        nu0s=3e10/(lam0s*1e-8)
        nuds=nu0s*dopbParam/3e5
        paramA=np.abs(gammas/(4.0*np.pi*nuds))
        xsectionsNearlinecenter=np.sqrt(np.pi)*((4.8032e-10)**2)*fs*lam0s/(9.110147603485839e-28*3e10*dopbParam*1e13)
        waveEdge=lam0s-3.0
        lamShifts=lam0s*velShifts/3e5
        
        firstPoint=waveEdge
        midPoint=lam0s
        endPoint=2*(midPoint-firstPoint)+firstPoint
        waveSymmetrical=np.linspace(firstPoint,endPoint,num=2*1000-1)
        waveOnesided=np.linspace(lam0s,waveEdge,num=1000)
        freqOnesided=3e10/(waveOnesided*1e-8)
        paramU=(freqOnesided-nu0s)/nuds
        
        xsectionsOnesided=xsectionsNearlinecenter*voigt.voigt(paramA,paramU)
        xsectionsOnesidedflipped=xsectionsOnesided[::-1]
        xsectionsSymmetrical=np.append(xsectionsOnesidedflipped[0:-1],xsectionsOnesided)
        deltaLam=np.max(waveSymmetrical)-np.min(waveSymmetrical)
        delLam=waveSymmetrical[1]-waveSymmetrical[0]
        nAll=np.round(deltaLam/delLam)
        waveAll=deltaLam*(np.arange(nAll)/(nAll-1))+waveSymmetrical[0]
        tauAll=np.interp(waveAll,waveSymmetrical+lamShifts,xsectionsSymmetrical*totN)
        
        return waveAll,tauAll
    
    def tauComponent(self,tauWave,lineCol,lineVel,lineDopb):
        #LyA, includes Hydrogen and Deuterium
        hwaveAll,htauAll=self.tauProfile(lineCol,lineVel,lineDopb,'h1')
        h1Tau=np.interp(tauWave,hwaveAll,htauAll)
        d1Col=np.log10((10.0**lineCol)*1.5e-5)
        dwaveAll,dtauAll=self.tauProfile(d1Col,lineVel,lineDopb,'d1')
        d1Tau=np.interp(tauWave,dwaveAll,dtauAll)
        
        totTau=h1Tau+d1Tau
        totISM=np.exp(-totTau)
            
        return totISM
    
    def selfrevComponent(self,revWave,revCent,revVrad,revFWHM,revAmpt):
        #model as absorbing optical depth profile, multiply with stellar profile
        revCent=revCent*((revVrad/3e5)+1) #redshifted line center
        revSigma=(revCent/3e5)*(revFWHM/2.3548) #Gaussian St. Dev
        
        selfrevProfile=Gaussian1D(amplitude=revAmpt,mean=revCent,stddev=revSigma) 
        revTau=1.0-selfrevProfile(revWave) 
        
        return revTau
    
    def loadAirglow(self,airgname):
        npdata=np.load(path+'//'+airgname+'.npy','rb')
        keylist=[]
        for keys in npdata:
            keylist.append(keys)
        return npdata,keylist
    
    def airglowComponent(self,waveGrid_A,shiftA,scaleA,lineA):
        #template for pure airglow, for all lines
        airgData,airgKeys=self.loadAirglow('Airglow Template Data 8-21-2021')
        
        if lineA==0:
            agInds=[0,1,2] #general LyA Template
        elif lineA==1:
            agInds=[3,4,5] #LP3 1327 LyA Template
        elif lineA==2:
            agInds=[6,7,8] #general OI 1302 Template
        elif lineA==5:
            agInds=[9,10,11] #general OI 1305 Template
        elif lineA==6:
            agInds=[12,13,14] #general OI 1306 Template
        
        airWave,airFlux,airErrr=airgData[airgKeys[agInds[0]]],airgData[airgKeys[agInds[1]]],airgData[airgKeys[agInds[2]]]
        airFlux_ss=scaleA*np.interp(waveGrid_A,airWave+shiftA,airFlux,left=0,right=0) #shifted & scaled airglow template
        airErrr_ss=scaleA*np.interp(waveGrid_A,airWave+shiftA,airErrr,left=0,right=0) #shifted & scaled airglow template error
        
        return airFlux_ss,airErrr_ss
    
    def totalModel(self,waveGrid_m,flux_m,error_m,lineCen_m,radialVel_m,ismVel_m,airShift_m,airScale_m,airLine_m):
        #define the model
        totModel=Model(starTemplate) 
        
        if self.isM:
            totParams=totModel.make_params(lineCen_s=lineCen_m,radialVel_s=radialVel_m,fwhmG_s=150.0,fwhmL_s=50.0,fluxAmp_s=-12.0,fwhmR_s=80.0,revAmpt_s=0.35,colDens_s=18.0,lineVel_s=ismVel_m,lineDopb_s=11.5,airShift_s=airShift_m,airScale_s=airScale_m,airLine_s=airLine_m)
        else:                                                                          #used to be 400 and 100
            totParams=totModel.make_params(lineCen_s=lineCen_m,radialVel_s=radialVel_m,fwhmG_s=150.0,fwhmL_s=50.0,fluxAmp_s=-12.0,fwhmR_s=80.0,revAmpt_s=0.50,colDens_s=18.0,lineVel_s=ismVel_m,lineDopb_s=11.5,airShift_s=airShift_m,airScale_s=airScale_m,airLine_s=airLine_m)
                                                    
        #set parameter limits or fix if unused                                                                                                   
        totParams['fwhmG_s'].min=0.0
        if self.fitIndex>0:
            totParams['fwhmG_s'].max=41.6 #value obtained from 55Cnc COS observations
        totParams['fluxAmp_s'].max=0.0 #may want to change to -8.0?
        totParams['airShift_s'].min=-1.5
        totParams['airShift_s'].max=1.5
        totParams['airScale_s'].min=0.0
        totParams['airScale_s'].max=10.0 #!!!change these airglow limits if the slider extrema are changed
        if self.fitIndex==0:
            totParams['fwhmL_s'].min=0.0 
            totParams['colDens_s'].min=17.1#17.53 #Table 2, Table 1 Values
            totParams['colDens_s'].max=19.1#19.11 #Table 1 Value
            totParams['lineVel_s'].min=-50.0 #Table 2 Value
            totParams['lineVel_s'].max=50.0 #Table 2 Value
        else:
            totParams['fwhmL_s'].vary=False
            totParams['colDens_s'].vary=False
            totParams['lineVel_s'].vary=False
        #set fixed parameters
        totParams['lineCen_s'].vary=False
        totParams['lineDopb_s'].vary=False
        totParams['revAmpt_s'].vary=False
        totParams['airLine_s'].vary=False
        #Handle the self reversal component
        if self.fitIndex>0:
            #O I triplet has no SR component
            totParams['fwhmR_s'].vary=False 
        else:
            #set the expression to fix the SR width
            totParams['fwhmR_s'].expr='0.47*fwhmG_s' #base this on width of the gaussian component
        #Handle the airglow mode
        if self.whichAir==1 or self.bootRunning: #manual mode
            totParams['airShift_s'].vary=False
            totParams['airScale_s'].vary=False
        
        #fitting results and other useful things
        totResult=totModel.fit(flux_m,totParams,waveGrid_s=waveGrid_m,weights=1.0/error_m,scale_covar=False)
        totBestfit=totResult.best_fit
        totReport=totResult.fit_report()
        totSqr=totResult.chisqr
        totRed=totResult.redchi
        totChi=[totSqr,totRed]
        totBic=totResult.bic
        
        return totBestfit,totResult.params,totReport,totChi,totBic

    def starComponents(self,waveGrid_s,lineCen_s,radialVel_s,fwhmG_s,fwhmL_s,fluxAmp_s,fwhmR_s,revAmpt_s,colDens_s,lineVel_s,lineDopb_s,airShift_s,airScale_s,airLine_s):
        #template for ISM  and self reversal attenuated voigt profile, for LyA
        strComp=self.stellarComponent(self.allWaveinfs[self.fitIndex],lineCen_s,radialVel_s,fwhmG_s,fwhmL_s,fluxAmp_s)
        if self.fitIndex==0: #LyA
            revComp=self.selfrevComponent(self.allWaveinfs[self.fitIndex],lineCen_s,radialVel_s,fwhmR_s,revAmpt_s)
            ismComp=self.tauComponent(self.allWaveinfs[self.fitIndex],colDens_s,lineVel_s,lineDopb_s)
            modFlux=(strComp*revComp)*ismComp
        else: #OI Triplet
            revComp=[1.0]*len(self.allWaveinfs[self.fitIndex]) #no need to compute these components
            ismComp=[1.0]*len(self.allWaveinfs[self.fitIndex]) #so just mult by 1.0 for the returns 
            modFlux=strComp
        airComp,errComp=self.airglowComponent(waveGrid_s,airShift_s,airScale_s,airLine_s)
        
        if len(modFlux)<len(self.lsfCOS):
            diffLen=len(self.lsfCOS)-len(modFlux)
            self.lsfTrunc=self.lsfCOS[int(np.ceil(diffLen/2.0)):-int(np.ceil(diffLen/2.0))]
            tocFlux=np.convolve(modFlux,self.lsfTrunc,mode='same')
        else:
            tocFlux=np.convolve(modFlux,self.lsfCOS,mode='same')
            
        preairFlux=np.interp(waveGrid_s,self.allWaveinfs[self.fitIndex],tocFlux)
        #flux on the (masked) instrument wavegrid, convolved with infinite kernel then interpolated onto data grid
        totalFlux=preairFlux+airComp
        
        return strComp,strComp*revComp,(strComp*revComp)*ismComp,airComp,errComp,preairFlux,totalFlux    

    def extractParam(self,parlist):
        #extract stellar parameters
        lcv=parlist['lineCen_s'].value
        lce=parlist['lineCen_s'].stderr
        rvv=parlist['radialVel_s'].value
        rve=parlist['radialVel_s'].stderr
        fgv=parlist['fwhmG_s'].value
        fge=parlist['fwhmG_s'].stderr 
        flv=parlist['fwhmL_s'].value
        fle=parlist['fwhmL_s'].stderr 
        fav=parlist['fluxAmp_s'].value
        fae=parlist['fluxAmp_s'].stderr
        
        #extract self reversal parameters
        frv=parlist['fwhmR_s'].value
        fre=parlist['fwhmR_s'].stderr
        rav=parlist['revAmpt_s'].value
        rae=parlist['revAmpt_s'].stderr
        
        #extract ISM parameters
        cdv=parlist['colDens_s'].value
        cde=parlist['colDens_s'].stderr
        ivv=parlist['lineVel_s'].value
        ive=parlist['lineVel_s'].stderr
        dbv=parlist['lineDopb_s'].value
        dbe=parlist['lineDopb_s'].stderr
        
        #extract airglow parameters
        ahv=parlist['airShift_s'].value
        ahe=parlist['airShift_s'].stderr
        acv=parlist['airScale_s'].value
        ace=parlist['airScale_s'].stderr
        
        all_param=[lcv,rvv,fgv,flv,fav,frv,rav,cdv,ivv,dbv,ahv,acv]
        all_parer=[lce,rve,fge,fle,fae,fre,rae,cde,ive,dbe,ahe,ace]
        
        return all_param,all_parer
    
    def powerConstruct(self,line_wave,line_cent,radi_velo,fwhm_gaus,fwhm_lrnz,flux_ampt,num_prf=1000,boot=False):
        #This function calculates the +/-1 sigma error for stellar profiles and their integrated fluxes, using self reversal where appropriate
        profl_holdr=np.zeros((num_prf,len(line_wave))) #placeholder array of all the stellar profiles
        iflux_holdr=np.zeros(num_prf) #placeholder array of all the integrated stellar profile fluxes
        for z in range(0,num_prf): 
        
            if isinstance(boot,bool): #if generating 1000 samples for LMFIT data
                #pull out a random value from a gaussian defined by the parameter's value and error, for each parameter
                radi_gauss=np.random.normal(radi_velo[0],radi_velo[1])
                gaus_gauss=np.random.normal(fwhm_gaus[0],fwhm_gaus[1])
                lrnz_gauss=np.random.normal(fwhm_lrnz[0],fwhm_lrnz[1])
                flux_gauss=np.random.normal(flux_ampt[0],flux_ampt[1])
            else: #if bootstrap samples have already been made with parameters
                #The bootstrap procedure already produced values, so there is no need to draw new ones from the parameter distrubutions
                radi_gauss=boot[1][z]
                gaus_gauss=boot[2][z]
                lrnz_gauss=boot[3][z]
                flux_gauss=boot[4][z]
            
            gaus_prof=self.stellarComponent(line_wave,line_cent,radi_gauss,gaus_gauss,lrnz_gauss,flux_gauss)
            revr_prof=self.selfrevComponent(line_wave,line_cent,radi_gauss,0.47*gaus_gauss,0.5)
            star_prof=gaus_prof*revr_prof
            
            #save this sample of the stellar profile    
            profl_holdr[z]=star_prof
            
            #get the total integrated flux of the stelar profile
            test_flux=self.integrateFlux(line_wave,star_prof)
            iflux_holdr[z]=test_flux
            
        #calculate the percentiles for +/-1sigma of the profiles and integrated fluxes, as well as the 50%   
        ub,mb,lb=np.percentile(profl_holdr,[84.135,50.000,15.865],axis=0) #upper, central, and lower wavebin profile flux value (+/-1 sigma)
        uf,mf,lf=np.percentile(iflux_holdr,[84.135,50.000,15.865],axis=0) #upper, central, and lower integrated flux value (+/-1 sigma)
        
        uu=ub-mb
        ll=mb-lb 
        ulErr=[uu,ll] #these are for the stellar profile
        uui=uf-mf 
        lli=mf-lf 
        uliErr=[uui,lli] #these are for the integrated stellar flux
        
        return ulErr,uliErr
    
    def auraBreak(self,line_wave,line_cent,radi_velo,fwhm_gaus,fwhm_lrnz,flux_ampt,coll_dens,vrad_isma,airg_shft,airg_scal,airg_indx,num_prf=1000,boot=False):
        #This function calculates the +/-1 sigma error for full model profiles
        profl_holdr=np.zeros((num_prf,len(line_wave))) #placeholder array of all full model profiles
        
        for z in range(0,num_prf): 
           
            if isinstance(boot,bool): #if generating 1000 samples for LMFIT data
                #pull out a random value from a gaussian defined by the parameter's value and error, for each parameter
                radi_gauss=np.random.normal(radi_velo[0],radi_velo[1])
                gaus_gauss=np.random.normal(fwhm_gaus[0],fwhm_gaus[1])
                lrnz_gauss=np.random.normal(fwhm_lrnz[0],fwhm_lrnz[1])
                flux_gauss=np.random.normal(flux_ampt[0],flux_ampt[1])
                ncol_gauss=np.random.normal(coll_dens[0],coll_dens[1])
                vism_gauss=np.random.normal(vrad_isma[0],vrad_isma[1])
                if self.whichAir==0:
                    airh_gauss=np.random.normal(airg_shft[0],airg_shft[1])
                    airc_gauss=np.random.normal(airg_scal[0],airg_scal[1])
            else: #if bootstrap samples have already been made with parameters
                #The bootstrap procedure already produced values, so there is no need to draw new ones from the parameter distrubutions
                radi_gauss=boot[1][z]
                gaus_gauss=boot[2][z]
                lrnz_gauss=boot[3][z]
                flux_gauss=boot[4][z]
                ncol_gauss=boot[7][z]
                vism_gauss=boot[8][z]
                if self.whichAir==0:
                    airh_gauss=boot[10][z]
                    airc_gauss=boot[11][z]
            
            if self.whichAir==0:
                Ztotl=starTemplate(line_wave,line_cent,radi_gauss,gaus_gauss,lrnz_gauss,flux_gauss,0.47*gaus_gauss,0.5,ncol_gauss,vism_gauss,11.5,airh_gauss,airc_gauss,airg_indx)
            else:
                Ztotl=starTemplate(line_wave,line_cent,radi_gauss,gaus_gauss,lrnz_gauss,flux_gauss,0.47*gaus_gauss,0.5,ncol_gauss,vism_gauss,11.5,airg_shft[0],airg_scal[0],airg_indx)                
            
            #save this sample of the full profile
            profl_holdr[z]=Ztotl
            
        #calculate the percentiles for +/-1sigma of the profiles, as well as the 50% (should be pretty close to the actual profile)     
        ub,mb,lb=np.percentile(profl_holdr,[84.135,50.000,15.865],axis=0) #upper flux value (1 sigma)
        
        uu=ub-mb
        ll=mb-lb 
        ulErr=[uu,ll]
        
        return ulErr
    
    def coreEnforcer(self,line_wave,line_flux,line_errr,num_prf=1000):
        #monte carlo of a given emission feature, generating a data point from each flux +/- error
        #generate N emission profiles, integrate them all, and get an integrated flux distribution
        samples=np.zeros((num_prf,len(line_wave)))
        sampflx=np.zeros((num_prf))
        for k in range(0,num_prf):
            indiv=[]
            for l in range(0,len(line_wave)):
                data_gauss=np.random.normal(line_flux[l],line_errr[l])
                indiv.append(data_gauss)
            samples[k]=indiv
            sampflx[k]=self.integrateFlux(line_wave,indiv) #integrate to a value, erg/cm^2/s
        ub,mb,lb=np.percentile(sampflx,[84.135,50.0,15.865])
        uu=ub-mb
        ll=mb-lb 
        uliErr=[uu,ll]
        return uliErr

    def runModel(self,twoOverride=False):
        if not twoOverride: #one part or the first half of two part
            self.waveModel=self.waveCOS[self.allLinemasks[self.fitIndex]]
            self.fluxModel=self.fluxCOS[self.allLinemasks[self.fitIndex]]
            self.errrModel=self.errrCOS[self.allLinemasks[self.fitIndex]]
            self.bestFit,self.fitParams,self.fitReport,self.fitChi,self.fitBic=self.totalModel(self.waveModel,self.fluxModel,self.errrModel,self.lineCenters[self.fitIndex],self.radVelocity,self.radISMVelocity,self.scaled_valH,self.scaled_valV,self.currentLine)
        else: #second half of two part
            self.waveModel=self.waveCOS[self.allLinemasks[self.fitIndex]]
            self.fluxModel=self.oneSpectrum[self.allLinemasks[self.fitIndex]] #one part recovered spectrum
            self.errrModel=self.oneError[self.allLinemasks[self.fitIndex]] #flux and airglow errors in quadrature    
            if self.whichAir!=1:
                self.whichAir=1 #temporatily swap into fixed airglow mode
                revert=True
            else:
                revert=False #if already in this mode, then no need to swap back to 1
            self.bestFit,self.fitParams,self.fitReport,self.fitChi,self.fitBic=self.totalModel(self.waveModel,self.fluxModel,self.errrModel,self.lineCenters[self.fitIndex],self.radVelocity,self.radISMVelocity,0.0,0.0,self.currentLine)
            if revert:
                self.whichAir=0 #go back to auto mode if this was the original setting
        pv,pe=self.extractParam(self.fitParams)
        if not twoOverride: #one part or the first half of two part
            self.stellarComp,self.selfreversalComp,self.ismComp,self.airglowComp,self.airglowErrr,self.convolvedComp,bestCopy=self.starComponents(self.waveModel,pv[0],pv[1],pv[2],pv[3],pv[4],pv[5],pv[6],pv[7],pv[8],pv[9],pv[10],pv[11],self.currentLine)
        else: #second half of two part
            pv[10]=self.airH
            pe[10]=self.airHerr
            pv[11]=self.airV
            pe[11]=self.airVerr #replace the 0.0s with the original airglow subtraction values
            self.stellarComp,self.selfreversalComp,self.ismComp,self.airglowComp,self.airglowErrr,self.convolvedComp,bestCopy=self.starComponents(self.waveModel,pv[0],pv[1],pv[2],pv[3],pv[4],pv[5],pv[6],pv[7],pv[8],pv[9],pv[10],pv[11],self.currentLine)
        self.pv=pv
        self.pe=pe #do the self version after to make the above component line shorter
        
        #calculate integrated fluxes
        if not twoOverride: #one part or the first half of two part
            tempRecov,tempError,tempRemov,tempRemer=self.recoverTrue(self.waveCOS,self.fluxCOS,self.errrCOS,self.pv[10]+self.allFitshifts[self.fitIndex],self.pv[11]*self.allFitscales[self.fitIndex],self.currentLine)
        else: #second half of two part
            tempRecov,tempError,tempRemov,tempRemer=self.oneSpectrum,self.oneError,self.oneRemove,self.oneRemerr
        self.intRecover=self.integrateFlux(self.waveCOS[self.allLinemasks[self.fitIndex]],tempRecov[self.allLinemasks[self.fitIndex]])
        self.intRecverr=self.coreEnforcer(self.waveCOS[self.allLinemasks[self.fitIndex]],tempRecov[self.allLinemasks[self.fitIndex]],tempError[self.allLinemasks[self.fitIndex]]) 
        self.intStellar=self.integrateFlux(self.allWaveinfs[self.fitIndex],self.selfreversalComp)
        try:
            self.intProferr,self.intStelerr=self.powerConstruct(self.allWaveinfs[self.fitIndex],self.pv[0],[self.pv[1],self.pe[1]],[self.pv[2],self.pe[2]],[self.pv[3],self.pe[3]],[self.pv[4],self.pe[4]])
            if not twoOverride: #one part or the first half of two part
                self.intBesterr=self.auraBreak(self.waveCOS[self.allLinemasks[self.fitIndex]],self.pv[0],[self.pv[1],self.pe[1]],[self.pv[2],self.pe[2]],[self.pv[3],self.pe[3]],[self.pv[4],self.pe[4]],[self.pv[7],self.pe[7]],[self.pv[8],self.pe[8]],[self.pv[10],self.pe[10]],[self.pv[11],self.pe[11]],self.currentLine)
            else: #second half of two part
                if self.whichAir!=1:
                    self.whichAir=1 #temporatily swap into fixed airglow mode
                    revert=True
                else:
                    revert=False #if already in this mode, then no need to swap back to 1
                self.intBesterr=self.auraBreak(self.waveCOS[self.allLinemasks[self.fitIndex]],self.pv[0],[self.pv[1],self.pe[1]],[self.pv[2],self.pe[2]],[self.pv[3],self.pe[3]],[self.pv[4],self.pe[4]],[self.pv[7],self.pe[7]],[self.pv[8],self.pe[8]],[0.0,0.0],[0.0,0.0],self.currentLine)
                if revert:
                    self.whichAir=0 #go back to auto mode if this was the original setting
        except:
            self.intProferr=[None,None]
            self.intStelerr=[None,None]
            self.intBesterr=[None,None]
        if self.readySTIS:
            #!!!may not need this anymore now that STIS data in the main GUI is updated in the STIS window
            maskSTIS=((self.currentRange[0]<=self.waveSTIS)&(self.waveSTIS<=self.currentRange[1]))
            try: #if no STIS data is present within the current range, cannot integrate
                self.intSTISdat=self.integrateFlux(self.waveSTIS[maskSTIS],self.fluxSTIS[maskSTIS]*self.sfSTIS[self.fitIndex]) #sfSTIS is not necessarily 1.0 here, need to scale
                self.intSTISerr=self.coreEnforcer(self.waveSTIS[maskSTIS],self.fluxSTIS[maskSTIS]*self.sfSTIS[self.fitIndex],self.errrSTIS[maskSTIS]*self.sfSTIS[self.fitIndex])
            except:
                self.intSTISdat=0.0
                self.intSTISerr=[False,False]       

    def runModel2(self):
        self.runModel()
        self.oneSpectrum,self.oneError,self.oneRemove,self.oneRemerr=self.recoverTrue(self.waveCOS,self.fluxCOS,self.errrCOS,self.pv[10]+self.allFitshifts[self.fitIndex],self.pv[11]*self.allFitscales[self.fitIndex],self.currentLine)
        self.airH=self.pv[10]
        self.airHerr=self.pe[10]
        self.airV=self.pv[11]
        self.airVerr=self.pe[11]
        self.runModel(twoOverride=True)        

    def readyFit(self):
        self.fitExists=True
        self.fitComponents.setDisabled(False)
        self.fitResults.setDisabled(False)
        self.openCutoff.setDisabled(False)
        self.removeAirglow.setDisabled(False)
        self.shiftInput.setDisabled(False)
        self.scaleInput.setDisabled(False)
        self.fitshiftInput.setDisabled(False)
        self.fitscaleInput.setDisabled(False)
        self.airShift.setDisabled(False)
        self.airScale.setDisabled(False)
        self.fitShift.setDisabled(False)
        self.fitScale.setDisabled(False)

    def onePart(self,selected):
        if selected:
            #user input, make sure change is okay
            if self.whichFit!=1:
                proceed=self.fitChanges()
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:
                self.whichFit=1 #one part=1
                
                if not self.fitExists and self.whichAir!=None:
                    self.readyFit()
                
                if self.fitExists:
                    self.runModel()
                    self.displayPlot()
            else:
                self.twoFit.toggled.disconnect()
                self.twoFit.toggle() #disconnect, reselect, reconnect
                self.twoFit.toggled.connect(self.twoPart)
                
    def twoPart(self,selected):
        if selected:
            #user input, make sure change is okay
            if self.whichFit!=2:
                proceed=self.fitChanges()
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:
                self.whichFit=2 #two part=2
                
                if not self.fitExists and self.whichAir!=None:
                    self.readyFit()
                
                if self.fitExists:
                    self.runModel2() #runs the model for 2 part fitting
                    self.displayPlot()
            else:
                self.oneFit.toggled.disconnect()
                self.oneFit.toggle() #disconnect, reselect, reconnect
                self.oneFit.toggled.connect(self.onePart)
            
    def autoMode(self,selected):
        if selected:
            #user input, make sure change is okay
            if self.whichFit!=2:
                proceed=self.fitChanges()
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:
                self.whichAir=0
    
                if not self.fitExists and self.whichFit!=None:
                    self.readyFit()
                    
                if self.fitExists:
                    if self.whichFit==1:
                        self.runModel()
                    else:
                        self.runModel2()
                    self.displayPlot()
            else:
               self.manualAirmode.toggled.disconnect()
               self.manualAirmode.toggle()
               self.manualAirmode.toggled.connect(self.manualMode)
               
               
    def manualMode(self,selected):
        if selected:
            #user input, make sure change is okay
            if self.whichFit!=2:
                proceed=self.fitChanges()
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:
                self.whichAir=1
                
                if not self.fitExists and self.whichFit!=None:
                    self.readyFit()
                
                if self.fitExists:
                    if self.whichFit==1:
                        self.runModel()
                    else:
                        self.runModel2()
                    self.displayPlot()
            else:
                self.autoAirmode.toggled.disconnect()
                self.autoAirmode.toggle()
                self.autoAirmode.toggled.connect(self.autoMode)
                
                
    def changeShift(self):
        try:
            temp_valH=int(round(float(self.shiftInput.text())*(1500.0/1.5))) #convert value back to -1500..1500 range
            isFloat=True
        except:
            notFloat=QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                isFloat=False
                
        if isFloat:
            if temp_valH>1500:
                unscaled_valH=1500
            elif temp_valH<-1500:
                unscaled_valH=-1500
            else:
                unscaled_valH=temp_valH
                
            #user input, make sure change is okay
            if round(unscaled_valH*(1.5/1500.0),3)!=self.scaled_valH:
                proceed=self.fitChanges()
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:
                self.scaled_valH=round(unscaled_valH*(1.5/1500.0),3)
                self.shiftInput.setText(str(self.scaled_valH))
                self.airShift.setValue(unscaled_valH)
                self.allShifts[self.fitIndex]=self.scaled_valH
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
                
                if self.whichFit==1:
                    self.runModel()
                else:
                    self.runModel2()
                self.displayPlot()
            else:
                self.shiftInput.setText(str(self.scaled_valH)) #return original value back to the input box
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
            
    def shiftDis(self):
        pass
    
    def shiftRec(self):
        self.airShift.valueChanged.connect(self.shiftApply)
        self.airShift.valueChanged.emit(self.airShift.value())
        
    def shiftApply(self,value):
        #user input, make sure change is okay
        if round(value*(1.5/1500.0),3)!=self.scaled_valH:
            proceed=self.fitChanges()
        else:
            proceed=False #if value is the same, then no need to recalculate the same fit
        if proceed:
            self.scaled_valH=round(value*(1.5/1500.0),3)
            self.shiftInput.setText(str(self.scaled_valH))
            self.allShifts[self.fitIndex]=self.scaled_valH
            self.airShift.valueChanged.disconnect()
            
            if not self.fitOverride:
                if self.whichFit==1:
                    self.runModel()
                else:
                    self.runModel2()
                self.displayPlot()
        else:
            self.airShift.valueChanged.disconnect() #disconnect before changing value, or else this function will be called on again
            self.airShift.setValue(int(round(self.scaled_valH*(1500.0/1.5)))) #return slider to original value
                
        
    def changeScale(self):
        try:
            temp_valV=int(round(float(self.scaleInput.text())*(1000.0/10.0))) #convert value back to 0..1000 range
            isFloat=True
        except:
            notFloat=QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                isFloat=False
                
        if isFloat:
            if temp_valV>1000:
                unscaled_valV=1000
            elif temp_valV<0:
                unscaled_valV=0
            else:
                unscaled_valV=temp_valV
                
            #user input, make sure change is okay
            if round(unscaled_valV*(10.0/1000.0),2)!=self.scaled_valV:
                proceed=self.fitChanges()
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:                
                self.scaled_valV=round(unscaled_valV*(10.0/1000.0),2)
                self.scaleInput.setText(str(self.scaled_valV))
                self.airScale.setValue(unscaled_valV)
                self.allScales[self.fitIndex]=self.scaled_valV
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
                
                if self.whichFit==1:
                    self.runModel()
                else:
                    self.runModel2()
                self.displayPlot()
            else:
                self.scaleInput.setText(str(self.scaled_valV)) #return original value back to the input box
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
            
    def scaleDis(self):
        pass
    
    def scaleRec(self):
        self.airScale.valueChanged.connect(self.scaleApply)
        self.airScale.valueChanged.emit(self.airScale.value())
        
    def scaleApply(self,value):
        #user input, make sure change is okay
        if round(value*(10.0/1000.0),2)!=self.scaled_valV:
            proceed=self.fitChanges()
        else:
            proceed=False #if value is the same, then no need to recalculate the same fit
        if proceed:
            self.scaled_valV=round(value*(10.0/1000.0),2)
            self.scaleInput.setText(str(self.scaled_valV))
            self.allScales[self.fitIndex]=self.scaled_valV
            self.airScale.valueChanged.disconnect()
            
            if not self.fitOverride:
                if self.whichFit==1:
                    self.runModel()
                else:
                    self.runModel2()
                self.displayPlot()
        else:
            self.airScale.valueChanged.disconnect() #disconnect before changing value, or else this function will be called on again
            self.airScale.setValue(int(round(self.scaled_valV*(1000.0/10.0)))) #return slider to original value   
            
        
    def applyShift(self):
        try:
            temp_valFH=int(round(float(self.fitshiftInput.text())*(1500.0/1.5))) #convert value back to -1500..1500 range
            isFloat=True
        except:
            notFloat=QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                isFloat=False
                
        if isFloat:
            if temp_valFH>1500:
                unscaled_valFH=1500
            elif temp_valFH<-1500:
                unscaled_valFH=-1500
            else:
                unscaled_valFH=temp_valFH
                
            #user input, make sure change is okay
            if round(unscaled_valFH*(1.5/1500.0),3)!=self.scaled_valFH:
                self.fitSlider=True #change came from a fit slider, best fit needs to be recalculated if residuals and/or RRCBB need to be reset
                proceed=self.fitChanges()
                self.fitSlider=False #whether proceed happens or not, reset this back to False to prevent it from running unnecessarily
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:
                self.scaled_valFH=round(unscaled_valFH*(1.5/1500.0),3)
                self.fitshiftInput.setText(str(self.scaled_valFH))
                self.fitShift.setValue(unscaled_valFH)
                self.allFitshifts[self.fitIndex]=self.scaled_valFH
                self.displayPlot()
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
            else:
                self.fitshiftInput.setText(str(self.scaled_valFH)) #return original value back to the input box
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
            
    
    def shiftFit(self,value):
        mousePos=self.cursor().pos().toPointF() #save cursor location
        #user input, make sure change is okay
        if round(value*(1.5/1500.0),3)!=self.scaled_valFH:
            self.fitSlider=True #change came from a fit slider, best fit needs to be recalculated if residuals and/or RRCBB need to be reset
            proceed=self.fitChanges()
            self.fitSlider=False #whether proceed happens or not, reset this back to False to prevent it from running unnecessarily
        else:
            proceed=False #if value is the same, then no need to recalculate the same fit
        if proceed:
            self.scaled_valFH=round(value*(1.5/1500.0),3)
            self.fitshiftInput.setText(str(self.scaled_valFH))
            self.allFitshifts[self.fitIndex]=self.scaled_valFH
            self.displayPlot()
        else:
            self.fitShift.valueChanged.disconnect() #disconnect to change things without calling this function again
            self.fitShift.setValue(int(round(self.scaled_valFH*(1500.0/1.5)))) #return slider to original value
            self.fitShift.valueChanged.connect(self.shiftFit) #reconnect
        
        #regardless of outcome, the proceed check messes up the slider handle appearance, need to simulate a mouse click to reset it            
        self.fitShift.valueChanged.disconnect()
        QtWidgets.QApplication.sendEvent(self.fitShift,QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonPress,mousePos,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.KeyboardModifier.NoModifier)) #press mouse
        QtWidgets.QApplication.sendEvent(self.fitShift,QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonRelease,mousePos,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.KeyboardModifier.NoModifier)) #release mouse
        self.fitShift.valueChanged.connect(self.shiftFit)
        
    def applyScale(self):
        try:
            temp_valFV=int(round(float(self.fitscaleInput.text())*(1000.0/10.0))) #convert value back to 0..1000 range
            isFloat=True
        except:
            notFloat=QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                isFloat=False
                
        if isFloat:
            if temp_valFV>1000:
                unscaled_valFV=1000
            elif temp_valFV<0:
                unscaled_valFV=0
            else:
                unscaled_valFV=temp_valFV
                
            #user input, make sure change is okay
            if round(unscaled_valFV*(10.0/1000.0),2)!=self.scaled_valFV:
                self.fitSlider=True #change came from a fit slider, best fit needs to be recalculated if residuals and/or RRCBB need to be reset
                proceed=self.fitChanges()
                self.fitSlider=False #whether proceed happens or not, reset this back to False to prevent it from running unnecessarily
            else:
                proceed=False #if value is the same, then no need to recalculate the same fit
            if proceed:
                self.scaled_valFV=round(unscaled_valFV*(10.0/1000.0),2)
                self.fitscaleInput.setText(str(self.scaled_valFV))
                self.fitScale.setValue(unscaled_valFV)
                self.allFitscales[self.fitIndex]=self.scaled_valFV
                self.displayPlot()
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
            else:
                self.fitscaleInput.setText(str(self.scaled_valFV)) #return original value back to the input box
                QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
            
    
    def scaleFit(self,value):
        mousePos=self.cursor().pos().toPointF() #save cursor location
        #user input, make sure change is okay
        if round(value*(10.0/1000.0),2)!=self.scaled_valFV:
            self.fitSlider=True #change came from a fit slider, best fit needs to be recalculated if residuals and/or RRCBB need to be reset
            proceed=self.fitChanges()
            self.fitSlider=False #whether proceed happens or not, reset this back to False to prevent it from running unnecessarily
        else:
            proceed=False #if value is the same, then no need to recalculate the same fit
        if proceed:
            self.scaled_valFV=round(value*(10.0/1000.0),2)
            self.fitscaleInput.setText(str(self.scaled_valFV))
            self.allFitscales[self.fitIndex]=self.scaled_valFV
            self.displayPlot()
        else:
            self.fitScale.valueChanged.disconnect() #disconnect to change things without calling this function again
            self.fitScale.setValue(int(round(self.scaled_valFV*(1000.0/10.0)))) #return slider to original value
            self.fitScale.valueChanged.connect(self.scaleFit) #reconnect
        
        #regardless of outcome, the proceed check messes up the slider handle appearance, need to simulate a mouse click to reset it
        self.fitScale.valueChanged.disconnect()
        QtWidgets.QApplication.sendEvent(self.fitScale,QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonPress,mousePos,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.KeyboardModifier.NoModifier)) #press mouse
        QtWidgets.QApplication.sendEvent(self.fitScale,QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonRelease,mousePos,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.KeyboardModifier.NoModifier)) #release mouse
        self.fitScale.valueChanged.connect(self.scaleFit)
        
    def toggleComponents(self):
        self.showComps=not(self.showComps)
        self.displayPlot()
        
    def showResults(self):
        dialogVals=resultWindow()
        dialogVals.exec()
        
    def readyCutoff(self):
        if not self.useCutoff[self.fitIndex]: #first time doing a cutoff for this line
            self.origMasks[self.fitIndex]=self.allLinemasks[self.fitIndex]
            self.origBests[self.fitIndex]=self.bestFit 
            self.origBerrs[self.fitIndex]=self.intBesterr
        #becuase this is a main window, I need to link it to the parent main window (or else it will close randomly)
        self.dialogResid=residualWindow()
        self.dialogResid.show()
        
    def integrateFlux(self,intWave,intFlux):
        #use simpsons rule to get the integrated flux
        if len(intWave)%2!=1: #if number of values isn't odd, can't use simpsons rule
            intWave=intWave[1:]
            intFlux=intFlux[1:] #cut off the first data point, usually left side has more ~zero flux
        #simpsons rule requires uniform wavegrid
        unifWave=np.linspace(intWave[0],intWave[-1],len(intWave))
        unifFlux=np.interp(unifWave,intWave,intFlux)
        #now integrate using simpsons rule
        integFlux=integrate.simps(unifFlux,unifWave) #y,x,integrates to a value, erg/cm^2/s
        return integFlux        
        
    def recoverTrue(self,dataWave,dataFlux,dataErrr,airgShift,airgScale,airgLine):
        airgData,airgKeys=self.loadAirglow('Airglow Template Data 8-21-2021')
        if airgLine==0:
            agInds=[0,1,2] #general LyA Template
        elif airgLine==1:
            agInds=[3,4,5] #LP3 1327 LyA Template
        elif airgLine==2:
            agInds=[6,7,8] #general OI 1302 Template
        elif airgLine==5:
            agInds=[9,10,11] #general OI 1305 Template
        elif airgLine==6:
            agInds=[12,13,14] #general OI 1306 Template   
        airgWave,airgFlux,airgErrr=airgData[airgKeys[agInds[0]]],airgData[airgKeys[agInds[1]]],airgData[airgKeys[agInds[2]]]
        
        #shift(interpolate) & scale template and its error onto the data's wavegrid
        airgWave=airgWave+airgShift
        airgFlux_ss=airgScale*np.interp(dataWave,airgWave,airgFlux,left=0.0,right=0.0)
        airgErrr_ss=airgScale*np.interp(dataWave,airgWave,airgErrr,left=0.0,right=0.0)
        #interpolate error array in same way as flux
        
        #obtain recovered spectrum
        recvFlux=dataFlux-airgFlux_ss #subtract airglow template
        recvErrr=np.sqrt(dataErrr**2.0+airgErrr_ss**2.0) #propagation of error
        
        return recvFlux,recvErrr,airgFlux_ss,airgErrr_ss
    
    def plotSubtraction(self):
        #2x2 plot for the chosen line and fit    
        fig,ax=plt.subplots(2,2,sharex=True,figsize=(16,9),layout='tight')
        fig.canvas.manager.set_window_title(self.plotStarname+'_'+self.lineLabel+'_Result')
        fig.supxlabel('Wavelength ($\AA$)',fontsize=20)
        fig.supylabel('Flux Density (erg $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)',fontsize=20)  
        fig.suptitle('Recovered Stellar Spectrum of '+self.plotStarname+': '+self.lineLabel,fontsize=26)
        plt.sca(ax[0][0])
        plt.plot(self.waveCOS[self.allLinemasks[self.fitIndex]],self.fluxCOS[self.allLinemasks[self.fitIndex]],label='COS Data',linewidth=2,color='C0')
        plt.fill_between(self.waveCOS[self.allLinemasks[self.fitIndex]],(self.fluxCOS+self.errrCOS)[self.allLinemasks[self.fitIndex]],(self.fluxCOS-self.errrCOS)[self.allLinemasks[self.fitIndex]],alpha=0.45)
        if self.whichFit==2:
            plt.plot(self.waveCOS,self.trueSpectrum,label='COS Recovered Spectrum',linewidth=2,linestyle='--',color='C9')
            plt.fill_between(self.waveCOS,self.trueSpectrum+self.propError,self.trueSpectrum-self.propError,alpha=0.45,color='C9')
        plt.plot(self.waveCOS[self.allLinemasks[self.fitIndex]]+self.allFitshifts[self.fitIndex],self.bestFit*self.allFitscales[self.fitIndex],label='Best Fit',linewidth=2,color='C1')
        try:
            plt.fill_between(self.waveCOS[self.allLinemasks[self.fitIndex]]+self.allFitshifts[self.fitIndex],(self.bestFit+self.intBesterr[0])*self.allFitscales[self.fitIndex],(self.bestFit-self.intBesterr[1])*self.allFitscales[self.fitIndex],color='C1',alpha=0.25)
        except:
            pass
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylim(-1e-16,1.2*self.ymaxU)
        ax[0][0].tick_params(labelsize=16)
        ax[0][0].set_title('Fit to COS Data',fontsize=18)
        ax[0][0].yaxis.set_ticks_position('both')
        ax[0][0].legend(fontsize=14)
        OoM00=ax[0][0].yaxis.get_offset_text()
        OoM00.set_size(18)
        plt.sca(ax[0][1])
        plt.plot(self.waveCOS[self.allLinemasks[self.fitIndex]]+self.allFitshifts[self.fitIndex],self.bestFit*self.allFitscales[self.fitIndex],label='Total Fit',linewidth=2,color='C1')
        try:
            plt.fill_between(self.waveCOS[self.allLinemasks[self.fitIndex]]+self.allFitshifts[self.fitIndex],(self.bestFit+self.intBesterr[0])*self.allFitscales[self.fitIndex],(self.bestFit-self.intBesterr[1])*self.allFitscales[self.fitIndex],color='C1',alpha=0.25)
        except:
            pass
        plt.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.stellarComp*self.allFitscales[self.fitIndex],label='Stellar Emission',linewidth=2,linestyle='-.',color='C5')
        if self.fitIndex<1:
            plt.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.selfreversalComp*self.allFitscales[self.fitIndex],label='SR Attn. Emission',linewidth=2,linestyle='-.',color='C6')
            try:
                plt.fill_between(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],(self.selfreversalComp+self.intProferr[0])*self.allFitscales[self.fitIndex],(self.selfreversalComp-self.intProferr[1])*self.allFitscales[self.fitIndex],color='C6',alpha=0.25)
            except:
                pass
            plt.plot(self.allWaveinfs[self.fitIndex]+self.allFitshifts[self.fitIndex],self.ismComp*self.allFitscales[self.fitIndex],label='SR+ISM Attn. Emission',linewidth=2,linestyle='-.',color='C3')
        plt.plot(self.waveCOS[self.allLinemasks[self.fitIndex]]+self.allFitshifts[self.fitIndex],self.airglowComp*self.allFitscales[self.fitIndex],label='Airglow Component',linewidth=2,linestyle='--',color='C4')
        plt.fill_between(self.waveCOS[self.allLinemasks[self.fitIndex]]+self.allFitshifts[self.fitIndex],(self.airglowComp+self.airglowErrr)*self.allFitscales[self.fitIndex],(self.airglowComp-self.airglowErrr)*self.allFitscales[self.fitIndex],alpha=0.45,color='C4')
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylim(-1e-16,1.2*self.ymaxU)
        ax[0][1].tick_params(labelsize=16)
        ax[0][1].set_title('Best Fit and Components',fontsize=18)
        ax[0][1].yaxis.set_ticks_position('both')
        ax[0][1].legend(fontsize=12)
        OoM01=ax[0][1].yaxis.get_offset_text()
        OoM01.set_size(18)
        plt.sca(ax[1][0])
        if self.readySTIS and not self.overrideSTIS:
            plt.plot(self.waveSTIS,self.fluxSTIS,label='STIS Spectrum',linewidth=2,color='C2')
            plt.fill_between(self.waveSTIS,self.fluxSTIS+self.errrSTIS,self.fluxSTIS-self.errrSTIS,alpha=0.25,color='C2')
            ax[1][0].set_title('Original STIS Data',fontsize=18)
            ax[1][0].legend(fontsize=14)
        else:
            ax[1][0].set_title('Original STIS Data (No STIS Data)',fontsize=18)
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylim(-1e-16,1.2*self.ymaxL)
        ax[1][0].tick_params(labelsize=16)
        ax[1][0].yaxis.set_ticks_position('both')
        OoM10=ax[1][0].yaxis.get_offset_text()
        OoM10.set_size(18)
        plt.xticks(rotation=30)
        plt.sca(ax[1][1])
        plt.plot(self.waveCOS,self.trueSpectrum,label='COS Recovered Spectrum',linewidth=2,color='C9')
        plt.fill_between(self.waveCOS,self.trueSpectrum+self.propError,self.trueSpectrum-self.propError,alpha=0.45,color='C9')
        if self.fitIndex<1:
            plt.plot(self.waveCOS[self.allLinemasks[self.fitIndex]],self.convolvedComp*self.allFitscales[self.fitIndex],label='Convolved SR+ISM Attn. Emission',linewidth=2,linestyle='--',color='C3')
        else:
            plt.plot(self.waveCOS[self.allLinemasks[self.fitIndex]],self.convolvedComp*self.allFitscales[self.fitIndex],label='Convolved Stellar Emission',linewidth=2,linestyle='--',color='C5')
        if self.readySTIS and not self.overrideSTIS:
            plt.plot(self.waveSTIS,self.fluxSTIS*self.sfSTIS[self.fitIndex],label='Scaled STIS Spectrum',linewidth=2,color='C2')
            plt.fill_between(self.waveSTIS,(self.fluxSTIS+self.errrSTIS)*self.sfSTIS[self.fitIndex],(self.fluxSTIS-self.errrSTIS)*self.sfSTIS[self.fitIndex],alpha=0.25,color='C2')
            ax[1][1].set_title('Comparison to STIS (scaled by '+str(round(self.sfSTIS[self.fitIndex],2))+')',fontsize=18)
        else:
            ax[1][1].set_title('Recovered Stellar Spectrum',fontsize=18)
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylim(-1e-16,1.2*self.ymaxL)
        ax[1][1].tick_params(labelsize=16)
        ax[1][1].yaxis.set_ticks_position('both')
        ax[1][1].legend(fontsize=12)
        OoM11=ax[1][1].yaxis.get_offset_text()
        OoM11.set_size(18)
        plt.xticks(rotation=30)
        plt.draw()
        
        #2x3 diagnostic plot for the chosen line and fit
        fig,ax=plt.subplots(3,2,sharex=True,figsize=(16,9),layout='tight')
        fig.canvas.manager.set_window_title(self.plotStarname+'_'+self.lineLabel+'_Diagnostic')
        fig.supxlabel('Wavelength ($\AA$)',fontsize=20)
        fig.suptitle(self.plotStarname+' Diagnostics: '+self.lineLabel,fontsize=26)
        plt.sca(ax[0][0])
        plt.semilogy(self.waveCOS,self.fluxNorm,linewidth=2.2,color='C0')
        plt.axhline(1.0,linewidth=2,linestyle='--',color='k')
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylim(9e-2,1.5e1)
        ax[0][0].tick_params(labelsize=16)
        ax[0][0].set_title('COS Flux Normalized by Error (log scale)',fontsize=18)
        ax[0][0].yaxis.set_ticks_position('both')
        OoM00=ax[0][0].yaxis.get_offset_text()
        OoM00.set_size(18)
        plt.sca(ax[0][1])
        plt.plot(self.waveCOS,self.fluxNorm,linewidth=2.2,color='C0')
        plt.axhline(1.0,linewidth=2,linestyle='--',color='k')
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylim(-1,11)
        ax[0][1].tick_params(labelsize=16)
        ax[0][1].set_title('COS Flux Normalized by Error (linear scale)',fontsize=18)
        ax[0][1].yaxis.set_ticks_position('both')
        OoM01=ax[0][1].yaxis.get_offset_text()
        OoM01.set_size(18)
        plt.sca(ax[1][0])
        if self.readySTIS and not self.overrideSTIS:
            plt.plot(self.waveSTIS,self.stistrueDiff,linewidth=2,color='C2')
            plt.ylim(1.2*self.diffMin,1.2*self.diffMax)
            ax[1][0].set_title('STIS - COS Residuals',fontsize=18)
        else:
            ax[1][0].set_title('STIS - COS Residuals (No STIS Data)',fontsize=18)
        plt.axhline(0.0,linewidth=2,linestyle='--',color='k')
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylabel('Flux Density (erg $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)',fontsize=14)
        ax[1][0].tick_params(labelsize=16)
        ax[1][0].yaxis.set_ticks_position('both')
        OoM10=ax[1][0].yaxis.get_offset_text()
        OoM10.set_size(18)
        plt.sca(ax[1][1])
        plt.plot(self.waveCOS,self.fluxfitDiff,linewidth=2,color='C1')
        plt.ylim(1.2*self.compMin,1.2*self.compMax)
        ax[1][1].set_title('COS - Best Fit Residuals',fontsize=18)
        plt.axhline(0.0,linewidth=2,linestyle='--',color='k')
        plt.xlim(self.currentRange[0],self.currentRange[1])
        plt.ylabel('Flux Density (erg $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)',fontsize=14)
        ax[1][1].tick_params(labelsize=16)
        ax[1][1].yaxis.set_ticks_position('both')
        OoM11=ax[1][1].yaxis.get_offset_text()
        OoM11.set_size(18)
        plt.sca(ax[2][0])
        if self.readySTIS and not self.overrideSTIS:
            plt.plot(self.waveSTIS,abs(self.diffNorm),linewidth=2,color='C2')
            plt.ylim(0,2.5)
            ax[2][0].set_title('STIS - COS Residuals Normalized by Error',fontsize=18)
        else:
            ax[2][0].set_title('STIS - COS Residuals Normalized by Error (No STIS Data)',fontsize=18)
        plt.axhline(1.0,linewidth=2,linestyle='--',color='k')
        plt.xlim(self.currentRange[0],self.currentRange[1])    
        ax[2][0].tick_params(labelsize=16)
        ax[2][0].yaxis.set_ticks_position('both')
        OoM20=ax[2][0].yaxis.get_offset_text()
        OoM20.set_size(18)
        plt.xticks(rotation=30)
        plt.sca(ax[2][1])
        plt.plot(self.waveCOS,abs(self.compNorm),linewidth=2,color='C1')
        plt.ylim(0,2.5)
        ax[2][1].set_title('COS - Best Fit Residuals Normalized by Error',fontsize=18)
        plt.axhline(1.0,linewidth=2,linestyle='--',color='k')
        plt.xlim(self.currentRange[0],self.currentRange[1])    
        ax[2][1].tick_params(labelsize=16)
        ax[2][1].yaxis.set_ticks_position('both')
        OoM21=ax[2][1].yaxis.get_offset_text()
        OoM21.set_size(18)
        plt.xticks(rotation=30)
        plt.draw()

    def getTrue(self):         
        if not self.airglowRemoved[self.fitIndex]:
            #first time airglow is being removed for the line
            if True in self.airglowRemoved: #if recovered profile has already been started for another line
                self.removeMethod='More' #subtract more airglow from the spectrum, for another emission line
            else:
                self.removeMethod='New' #subtract airglow from the spectrum for the first time
            self.airglowRemoved[self.fitIndex]=True
            self.removalPlots.setDisabled(False)
            self.openBootstrap.setDisabled(False)
            self.saveData.setDisabled(False)
        else:
            #check if the current line has had airglow removed already
            if self.airglowRemoved[self.fitIndex]:
                dialogSpec=spectrumWindow()
                dialogSpec.exec()

        if self.removeMethod=='New': #remove from the original COS data
            if self.whichFit==1: #one part fitting
                self.trueSpectrum,self.propError,self.removeFlux,self.airglError=self.recoverTrue(self.waveCOS,self.fluxCOS,self.errrCOS,self.pv[10]+self.allFitshifts[self.fitIndex],self.pv[11]*self.allFitscales[self.fitIndex],self.currentLine)
            else: #two part fitting
                self.trueSpectrum,self.propError,self.removeFlux,self.airglError=self.oneSpectrum,self.oneError,self.oneRemove,self.oneRemerr
        elif self.removeMethod=='More':
            self.trueSpectrum,self.propError,self.removeFlux,self.airglError=self.recoverTrue(self.waveCOS,self.trueSpectrum,self.propError,self.pv[10]+self.allFitshifts[self.fitIndex],self.pv[11]*self.allFitscales[self.fitIndex],self.currentLine)
        elif self.removeMethod=='Cancel':
            pass #do not remove airglow if cancel is selected in the dialog window
        
        if self.removeMethod!='Cancel': #do not plot subtraction plots if airglow was not supposed to be subtracted
            #setup Y limits for the 2x2 plot
            self.ymaxU=max(np.max(self.fluxCOS[self.allLinemasks[self.fitIndex]]),np.max(self.bestFit))
            
            if self.fitIndex!=0 and self.whichSTIS=='G':
                self.overrideSTIS=True #No OI data for G140M
            elif not self.readySTIS:
                self.overrideSTIS=True
            else:
                self.overrideSTIS=False
                
            if self.readySTIS and not self.overrideSTIS:
                stisMask=((self.currentRange[0]<=self.waveSTIS)&(self.waveSTIS<=self.currentRange[1]))
                self.ymaxL=max(np.max((self.fluxSTIS*self.sfSTIS[self.fitIndex])[stisMask]),np.max(self.fluxSTIS[stisMask]),np.max(self.trueSpectrum[self.allLinemasks[self.fitIndex]]))
            else:
                self.ymaxL=np.max(self.trueSpectrum[self.allLinemasks[self.fitIndex]])
                
            #setup for diagnostics
            #need to figure out how to deal with dividing by zero in fluxNorm and compNorm, likely also diffNorm
            self.fluxNorm=self.fluxCOS/self.errrCOS
            self.bestInterp=np.interp(self.waveCOS,self.waveCOS[self.allLinemasks[self.fitIndex]]+self.allFitshifts[self.fitIndex],self.bestFit,left=0.0,right=0.0)
            if self.whichFit==1:
                self.fluxfitDiff=self.fluxCOS-self.bestInterp
                self.compNorm=self.fluxfitDiff/self.errrCOS
            else:
                self.fluxfitDiff=self.trueSpectrum-self.bestInterp
                self.compNorm=self.fluxfitDiff/self.propError
            self.compMin=np.min(self.fluxfitDiff[self.allLinemasks[self.fitIndex]])
            self.compMax=np.max(self.fluxfitDiff[self.allLinemasks[self.fitIndex]])
            if self.readySTIS and not self.overrideSTIS:
                self.trueInterp=np.interp(self.waveSTIS,self.waveCOS,self.trueSpectrum)
                self.stistrueDiff=(self.fluxSTIS*self.sfSTIS[self.fitIndex])-self.trueInterp
                self.diffNorm=self.stistrueDiff/(self.errrSTIS*self.sfSTIS[self.fitIndex])
                self.diffMin=np.min(self.stistrueDiff[stisMask])
                self.diffMax=np.max(self.stistrueDiff[stisMask])
                
            self.plotSubtraction()
            
            #save components to their savefile prep arrays
            self.finalLinemasks[self.fitIndex]=self.allLinemasks[self.fitIndex] #keep what was used for the subtraction
            self.finalWaveinfs[self.fitIndex]=self.allWaveinfs[self.fitIndex] #keep what was used for the subtraction
            self.finalStellar[self.fitIndex]=self.stellarComp       
            self.finalStelerr[self.fitIndex]=self.intProferr #upper and lower
            self.finalReversal[self.fitIndex]=self.selfreversalComp
            self.finalISM[self.fitIndex]=self.ismComp
            self.finalAirglow[self.fitIndex]=self.removeFlux
            self.finalAirgerr[self.fitIndex]=self.airglError
            self.finalParams[self.fitIndex]=[self.pv,self.pe]
            self.finalBest[self.fitIndex]=self.bestFit
            self.finalBesterr[self.fitIndex]=self.intBesterr #upper and lower
            self.finalIfluxrecv[self.fitIndex]=self.intRecover
            self.finalIfluxrerr[self.fitIndex]=self.intRecverr #upper and lower
            self.finalIfluxstel[self.fitIndex]=self.intStellar
            self.finalIfluxserr[self.fitIndex]=self.intStelerr #upper and lower
            if self.readySTIS and not self.overrideSTIS:
                self.finalIfluxstis[self.fitIndex]=self.intSTISdat
                self.finalIfluxster[self.fitIndex]=self.intSTISerr #upper and lower
                self.finalSTISscale[self.fitIndex]=self.sfSTIS[self.fitIndex]
            self.finalFitmode[self.fitIndex]=self.whichFit
            self.finalAirmode[self.fitIndex]=self.whichAir
            self.finalCutoffs[self.fitIndex]=self.allCutoffs[self.fitIndex]
            self.finalUshifts[self.fitIndex]=self.allShifts[self.fitIndex]
            self.finalUscales[self.fitIndex]=self.allScales[self.fitIndex]
            self.finalUfitshifts[self.fitIndex]=self.allFitshifts[self.fitIndex]
            self.finalUfitscales[self.fitIndex]=self.allFitscales[self.fitIndex]
            self.finalRadial[self.fitIndex]=self.radVelocity
            self.finalRadism[self.fitIndex]=self.radISMVelocity
            #all bootstrap related final saves are handled internally in the bootstrap class
        
    def readyRRCBB(self):
        self.dialogBoot=bootstrapWindow()
        self.dialogBoot.exec() 
        
    def formatParams(self,paralists,rcberrors):
        paramNames=['Line Center (Å)','Stellar Radial Velocity (km/s)','Gaussian FWHM (km/s)','Lorentzian FWHM (km/s)','Flux Amplitude (10^x)','Self Reversal FWHM (km/s)','Self Reversal Depth Fraction','Column Density (10^x cm^-2)','ISM Radial Velocity (km/s)','Doppler b Parameter (km/s)','Shift (Å)','Scale']
        
        #LyA parameters
        try:
            paramsLyA,errorsLyA=paralists[0]
            if np.all(errorsLyA)==None:
                errorpLyA=['N/A']*12
                errormLyA=['N/A']*12
            else:
                errorpLyA=errormLyA=errorsLyA
        except:
            paramsLyA=['N/A']*12 #12 is the number of parameters, as all fits report 12 now
            errorpLyA=['N/A']*12
            errormLyA=['N/A']*12
            
        if self.finalEmethod[0]=='RRCBB': #override with RRCBB errors
            errorpLyA=rcberrors[0][0]
            errormLyA=rcberrors[0][1]
            if self.finalFitmode[0]==2: #reoverride the airglow errors in 2 part fitting
                errorpLyA[10]=errormLyA[10]=errorsLyA[10]
                errorpLyA[11]=errormLyA[11]=errorsLyA[11]
            
        #OI 1302 parameters
        try:
            paramsOI2,errorsOI2=paralists[1]
            if np.all(errorsOI2)==None:
                errorpOI2=['N/A']*12
                errormOI2=['N/A']*12
            else:
                errorpOI2=errormOI2=errorsOI2
        except:
            paramsOI2=['N/A']*12
            errorpOI2=['N/A']*12
            errormOI2=['N/A']*12
            
        if self.finalEmethod[1]=='RRCBB': #override with RRCBB errors
            errorpOI2=rcberrors[1][0]
            errormOI2=rcberrors[1][1]
            if self.finalFitmode[1]==2: #reoverride the airglow errors in 2 part fitting
                errorpOI2[10]=errormOI2[10]=errorsOI2[10]
                errorpOI2[11]=errormOI2[11]=errorsOI2[11]
            
        #OI 1305 parameters
        try:
            paramsOI5,errorsOI5=paralists[2]
            if np.all(errorsOI5)==None:
                errorpOI5=['N/A']*12
                errormOI5=['N/A']*12
            else:
                errorpOI5=errormOI5=errorsOI5
        except:
            paramsOI5=['N/A']*12
            errorpOI5=['N/A']*12
            errormOI5=['N/A']*12
            
        if self.finalEmethod[2]=='RRCBB': #override with RRCBB errors
            errorpOI5=rcberrors[2][0]
            errormOI5=rcberrors[2][1]
            if self.finalFitmode[2]==2: #reoverride the airglow errors in 2 part fitting
                errorpOI5[10]=errormOI5[10]=errorsOI5[10]
                errorpOI5[11]=errormOI5[11]=errorsOI5[11]
            
        #OI 1306 parameters
        try:
            paramsOI6,errorsOI6=paralists[3]
            if np.all(errorsOI6)==None:
                errorpOI6=['N/A']*12
                errormOI6=['N/A']*12
            else:
                errorpOI6=errormOI6=errorsOI6
        except:
            paramsOI6=['N/A']*12
            errorpOI6=['N/A']*12
            errormOI6=['N/A']*12
            
        if self.finalEmethod[3]=='RRCBB': #override with RRCBB errors
            errorpOI6=rcberrors[3][0]
            errormOI6=rcberrors[3][1]
            if self.finalFitmode[3]==2: #reoverride the airglow errors in 2 part fitting
                errorpOI6[10]=errormOI6[10]=errorsOI6[10]
                errorpOI6[11]=errormOI6[11]=errorsOI6[11]
            
        formatLines=[]
        for x in range(0,12): #they will all have length 12 no matter what
            formatLines.append([paramNames[x],paramsLyA[x],errorpLyA[x],errormLyA[x],paramsOI2[x],errorpOI2[x],errormOI2[x],paramsOI5[x],errorpOI5[x],errormOI5[x],paramsOI6[x],errorpOI6[x],errormOI6[x]])
        
        return formatLines
    
    def formatInf(self,infxvals,infarray):
        if isinstance(infarray,list):
            infarray=np.array(infarray)
        if infarray.ndim==1:
            infarray=np.array([infarray])
            revert=True #need to return a 1 dim list 
        else:
            revert=False
        interpInf=[]
        for i in range(0,len(infarray)):
            try:
                tmparray=np.interp(self.waveCOS,infxvals,infarray[i],left=0,right=0)
                interpInf.append(tmparray)
            except:
                interpInf.append(np.zeros(len(self.waveCOS))) #if None reported (or some other issue), put zero for the error
        if revert:
            return np.array(interpInf[0])
        else:
            return np.array(interpInf)
    
    def saveBoot(self,bootD,savename):
        npf=open(savename+'.npy','wb') #w=write
        np.savez(npf,**bootD) #the ** passes in the keywords of the dictionary and the data inside each key
        npf.close()
    
    def saveTrue(self):
        saveReady=True
        
        #construct a simple messagebox with a check mark (makinga  separate QDialog in QtDesigner would have been overkill)
        saveDialog=QtWidgets.QMessageBox()
        saveDialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
        saveDialog.setWindowTitle('Do you want to continue?')
        saveDialog.setText('Do you want to save this recovered spectrum?\n Press No to continue creating the recovered spectrum.')
        saveDialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
        saveCheck=QtWidgets.QCheckBox('Save Individual RRCBB Samples?')
        saveCheck.setToolTip('File size can be large (100s of MB), unchecking will still save RRCBB error values')
        if True in self.bootDone:
            saveCheck.setChecked(True) #checked by default if at least one bootstrap was performed
        saveDialog.setCheckBox(saveCheck)
        saveValue=saveDialog.exec()

        if saveValue==QtWidgets.QMessageBox.StandardButton.No:
            saveReady=False
        
        if saveReady:
            #set any Falses in the final arrays to zero arrays, arrays of zero arrays, or N/As, depending on the type of final array
            self.finalStellar=[np.zeros(len(self.waveCOS)) if isinstance(self.finalStellar[t],bool) else self.formatInf(self.finalWaveinfs[t],self.finalStellar[t]) for t in range(0,len(self.finalStellar))]
            self.finalStelerr=[[np.zeros(len(self.waveCOS))]*2 if isinstance(self.finalStelerr[cc],bool) else self.formatInf(self.finalWaveinfs[cc],self.finalStelerr[cc]) for cc in range(0,len(self.finalStelerr))]
            self.finalReversal=[np.zeros(len(self.waveCOS)) if isinstance(self.finalReversal[d],bool) else self.formatInf(self.finalWaveinfs[d],self.finalReversal[d]) for d in range(0,len(self.finalReversal))]
            self.finalISM=[np.zeros(len(self.waveCOS)) if isinstance(self.finalISM[u],bool) else self.formatInf(self.finalWaveinfs[u],self.finalISM[u]) for u in range(0,len(self.finalISM))]
            self.finalAirglow=[np.zeros(len(self.waveCOS)) if isinstance(v,bool) else v for v in self.finalAirglow]
            self.finalAirgerr=[np.zeros(len(self.waveCOS)) if isinstance(w,bool) else w for w in self.finalAirgerr]
            self.finalBest=[np.zeros(len(self.waveCOS)) if isinstance(self.finalBest[bb],bool) else self.formatInf(self.waveCOS[self.finalLinemasks[bb]],self.finalBest[bb]) for bb in range(0,len(self.finalBest))]
            self.finalBesterr=[[np.zeros(len(self.waveCOS))]*2 if isinstance(self.finalBesterr[dd],bool) else self.formatInf(self.waveCOS[self.finalLinemasks[dd]],self.finalBesterr[dd]) for dd in range(0,len(self.finalBesterr))]            
            self.finalIfluxrecv=['N/A' if isinstance(e,bool) else e for e in self.finalIfluxrecv]
            self.finalIfluxrerr=[[0,0] if isinstance(ii,bool) else ii for ii in self.finalIfluxrerr]
            self.finalIfluxstel=['N/A' if isinstance(ee,bool) else ee for ee in self.finalIfluxstel]
            self.finalIfluxserr=[[0,0] if isinstance(ff,bool) else ff for ff in self.finalIfluxserr]
            self.finalIfluxstis=['N/A' if isinstance(jj,bool) else jj for jj in self.finalIfluxstis]
            self.finalIfluxster=[[0,0] if isinstance(kk,bool) else kk for kk in self.finalIfluxster]
            self.finalSTISscale=['N/A' if isinstance(oo,bool) else oo for oo in self.finalSTISscale]
            self.finalFitmode=['N/A' if isinstance(aa,bool) else aa for aa in self.finalFitmode]
            self.finalAirmode=['N/A' if isinstance(f,bool) else f for f in self.finalAirmode]
            self.finalEmethod=['RRCBB' if isinstance(g,bool) and g==True else 'LMFIT' for g in self.finalEmethod]
            self.finalCutoffs=['N/A' if isinstance(h,bool) else h for h in self.finalCutoffs]
            self.finalUshifts=['N/A' if isinstance(z,bool) else z for z in self.finalUshifts]
            self.finalUscales=['N/A' if isinstance(a,bool) else a for a in self.finalUscales]
            self.finalUfitshifts=['N/A' if isinstance(b,bool) else b for b in self.finalUfitshifts]
            self.finalUfitscales=['N/A' if isinstance(k,bool) else k for k in self.finalUfitscales]
            self.finalRadial=['N/A' if isinstance(c,bool) else c for c in self.finalRadial]
            self.finalRadism=['N/A' if isinstance(l,bool) else l for l in self.finalRadism]
            self.finalOptlen=['N/A' if isinstance(tt,bool) else tt for tt in self.finalOptlen]
            self.finalNumblk=['N/A' if isinstance(uu,bool) else uu for uu in self.finalNumblk]
            self.finalNumpar=['N/A' if isinstance(vv,bool) else vv for vv in self.finalNumpar]
            self.finalNumsmp=['N/A' if isinstance(ww,bool) else ww for ww in self.finalNumsmp]
            self.finalDeltaT=['N/A' if isinstance(xx,bool) else xx for xx in self.finalDeltaT]
            self.finalRCBfit=[[np.zeros(len(self.waveCOS))]*len(self.finalNumsmp[yy]) if isinstance(self.finalRCBfit[yy],bool) else self.formatInf(self.waveCOS[self.finalLinemasks[yy]],self.finalRCBfit[yy]) for yy in range(0,len(self.finalRCBfit))]
            self.finalRCBste=[[np.zeros(len(self.waveCOS))]*len(self.finalNumsmp[zz]) if isinstance(self.finalRCBste[zz],bool) else self.formatInf(self.finalWaveinfs[zz],self.finalRCBste[zz]) for zz in range(0,len(self.finalRCBste))]
            #these come after checking finalNumsmp to avoid using the length of a false
            #get parameters formatted to be put into the csv easily
            formattedParams=self.formatParams(self.finalParams,self.finalRCBpae)
            #for now, make new variables with LMFIT flux errors or RRCBB flux errors if it was done for the line 
            self.bestproferr=[]
            self.starproferr=[]
            self.starerr=[]
            for gg in range(0,len(self.finalEmethod)):
                if self.finalEmethod[gg]=='RRCBB':
                    self.bestproferr.append(self.formatInf(self.waveCOS[self.finalLinemasks[gg]],self.finalRCBbfe[gg]))
                    self.starproferr.append(self.formatInf(self.finalWaveinfs[gg],np.array(self.finalRCBsce[gg])))
                    self.starerr.append(self.finalRCBsie[gg])
                else:
                    if len(self.finalStelerr[gg])==2: 
                        self.bestproferr.append(self.finalBesterr[gg])
                        self.starproferr.append(self.finalStelerr[gg])
                    else:
                        self.bestproferr.append([self.finalBesterr[gg],self.finalBesterr[gg]])
                        self.starproferr.append([self.finalStelerr[gg],self.finalStelerr[gg]])
                    self.starerr.append(self.finalIfluxserr[gg]) 
            #format some of the values for the additional data csv
            self.finalFitmode=['One Part' if aa==1 else aa for aa in self.finalFitmode]
            self.finalFitmode=['Two Part' if aa==2 else aa for aa in self.finalFitmode]
            self.finalAirmode=['Automatic' if f==0 else f for f in self.finalAirmode]
            self.finalAirmode=['Manual' if f==1 else f for f in self.finalAirmode]
            
            #save two CSV files (use xlsxwriter in the future to save two tabs into an excel spreadsheet)
            name2=QtWidgets.QFileDialog.getSaveFileName(self,'Save File','','CSV(*.csv)')
            if name2[0][-4:]!='.csv': #some linux distributions do not append the file extension (may be related to no 'All' option)
                name2=(name2[0]+'.csv',name2[1]) #tuple is immutable, can't append .csv to name2[0]
            if name2[0]: #utf-16 to display Å and α properly, and newline='' fixes line skipping in the csv, '\t' allows for csv.write to work with utf-16
                with open(name2[0],mode='w',encoding='utf-16',newline='') as dataline: #wavelength, flux, and error arrays
                    line=csv.writer(dataline, delimiter='\t')
                    line.writerow(['Wavelength (Å)','Recovered Flux (erg cm^-2 s^-1 Å^-1)','Recovered Error (erg cm^-2 s^-1 Å^-1)','Lyα Modeled Stellar Flux (erg cm^-2 s^-1 Å^-1)','Lyα Modeled SR Attenuated Flux (erg cm^-2 s^-1 Å^-1)','Lyα Modeled SR Attenuated Flux Error+ (erg cm^-2 s^-1 Å^-1)','Lyα Modeled SR Attenuated Flux Error- (erg cm^-2 s^-1 Å^-1)', 'Lyα Modeled SR + ISM Attenuated Flux (erg cm^-2 s^-1 Å^-1)','Lyα Removed Airglow Flux (erg cm^-2 s^-1 Å^-1)','Lyα Removed Airglow Error (erg cm^-2 s^-1 Å^-1)','Lyα Best Fit (erg cm^-2 s^-1 Å^-1)','Lyα Best Fit Error+ (erg cm^-2 s^-1 Å^-1)','Lyα Best Fit Error- (erg cm^-2 s^-1 Å^-1)','OI 1302 Modeled Stellar Flux (erg cm^-2 s^-1 Å^-1)','OI 1302 Modeled SR Attenuated Flux (erg cm^-2 s^-1 Å^-1)','OI 1302 Modeled SR Attenuated Flux Error+ (erg cm^-2 s^-1 Å^-1)','OI 1302 Modeled SR Attenuated Flux Error- (erg cm^-2 s^-1 Å^-1)','OI 1302 Modeled SR + ISM Attenuated Flux (erg cm^-2 s^-1 Å^-1)','OI 1302 Removed Airglow Flux (erg cm^-2 s^-1 Å^-1)','OI 1302 Removed Airglow Error (erg cm^-2 s^-1 Å^-1)','OI 1302 Best Fit (erg cm^-2 s^-1 Å^-1)','OI 1302 Best Fit Error+ (erg cm^-2 s^-1 Å^-1)','OI 1302 Best Fit Error- (erg cm^-2 s^-1 Å^-1)','OI 1305 Modeled Stellar Flux (erg cm^-2 s^-1 Å^-1)','OI 1305 Modeled SR Attenuated Flux (erg cm^-2 s^-1 Å^-1)','OI 1305 Modeled SR Attenuated Flux Error+ (erg cm^-2 s^-1 Å^-1)','OI 1305 Modeled SR Attenuated Flux Error- (erg cm^-2 s^-1 Å^-1)','OI 1305 Modeled SR + ISM Attenuated Flux (erg cm^-2 s^-1 Å^-1)','OI 1305 Removed Airglow Flux (erg cm^-2 s^-1 Å^-1)','OI 1305 Removed Airglow Error (erg cm^-2 s^-1 Å^-1)','OI 1305 Best Fit (erg cm^-2 s^-1 Å^-1)','OI 1305 Best Fit Error+ (erg cm^-2 s^-1 Å^-1)','OI 1305 Best Fit Error- (erg cm^-2 s^-1 Å^-1)','OI 1306 Modeled Stellar Flux (erg cm^-2 s^-1 Å^-1)','OI 1306 Modeled SR Attenuated Flux (erg cm^-2 s^-1 Å^-1)','OI 1306 Modeled SR Attenuated Flux Error+ (erg cm^-2 s^-1 Å^-1)','OI 1306 Modeled SR Attenuated Flux Error- (erg cm^-2 s^-1 Å^-1)','OI 1306 Modeled SR + ISM Attenuated Flux (erg cm^-2 s^-1 Å^-1)','OI 1306 Removed Airglow Flux (erg cm^-2 s^-1 Å^-1)','OI 1306 Removed Airglow Error (erg cm^-2 s^-1 Å^-1)','OI 1306 Best Fit (erg cm^-2 s^-1 Å^-1)','OI 1306 Best Fit Error+ (erg cm^-2 s^-1 Å^-1)','OI 1306 Best Fit Error- (erg cm^-2 s^-1 Å^-1)'])
                    for s in range(0,len(self.waveCOS)):
                        line.writerow([self.waveCOS[s],self.trueSpectrum[s],               self.propError[s],                      self.finalStellar[0][s],                         self.finalReversal[0][s],                              self.starproferr[0][0][s],                                      self.starproferr[0][1][s],                                     self.finalISM[0][s],                                         self.finalAirglow[0][s],                         self.finalAirgerr[0][s],                          self.finalBest[0][s],                self.bestproferr[0][0][s],                   self.bestproferr[0][1][s],                   self.finalStellar[1][s],                             self.finalReversal[1][s],                                  self.starproferr[1][0][s],                                         self.starproferr[1][1][s],                                         self.finalISM[1][s],                                             self.finalAirglow[1][s],                             self.finalAirgerr[1][s],                              self.finalBest[1][s],                    self.bestproferr[1][0][s],                       self.bestproferr[1][1][s],                       self.finalStellar[2][s],                             self.finalReversal[2][s],                                  self.starproferr[2][0][s],                                         self.starproferr[2][1][s],                                         self.finalISM[2][s],                                             self.finalAirglow[2][s],                             self.finalAirgerr[2][s],                              self.finalBest[2][s],                    self.bestproferr[2][0][s],                       self.bestproferr[2][1][s],                       self.finalStellar[3][s],                             self.finalReversal[3][s],                                  self.starproferr[3][0][s],                                         self.starproferr[3][1][s],                                         self.finalISM[3][s],                                             self.finalAirglow[3][s],                             self.finalAirgerr[3][s],                             self.finalBest[3][s],                     self.bestproferr[3][0][s],                       self.bestproferr[3][1][s]])
                with open(name2[0][:-4]+'-Additional Data.csv',mode='w',encoding='utf-16',newline='') as dataline: #best fit parameters and user inputs
                    line=csv.writer(dataline,delimiter='\t')
                    line.writerow(['Parameter','Lyα Value','Lyα Value Error+','Lyα Value Error-','OI 1302 Value','OI 1302 Value Error+','OI 1302 Value Error-','OI 1305 Value','OI 1305 Value Error+','OI 1305 Value Error-','OI 1306 Value','OI 1306 Value Error+','OI 1306 Value Error-'])
                    for y in range(0,len(formattedParams)):
                        line.writerow(formattedParams[y])
                    line.writerow(['','','']) #separator
                    line.writerow(['Integrated Fluxes','Lyα Flux','Lyα Flux Error+','Lyα Flux Error-','OI2 Flux','OI2 Flux Error+','OI2 Flux Error-','OI5 Flux','OI5 Flux Error+','OI5 Flux Error-','OI6 Flux','OI6 Flux Error+','OI6 Flux Error-']) 
                    line.writerow(['Stellar Flux',self.finalIfluxstel[0],self.starerr[0][0],self.starerr[0][1],self.finalIfluxstel[1],self.starerr[1][0],self.starerr[1][1],self.finalIfluxstel[2],self.starerr[2][0],self.starerr[2][1],self.finalIfluxstel[3],self.starerr[3][0],self.starerr[3][1]])
                    line.writerow(['Recovered Flux',self.finalIfluxrecv[0],self.finalIfluxrerr[0][0],self.finalIfluxrerr[0][1],self.finalIfluxrecv[1],self.finalIfluxrerr[1][0],self.finalIfluxrerr[1][1],self.finalIfluxrecv[2],self.finalIfluxrerr[2][0],self.finalIfluxrerr[2][1],self.finalIfluxrecv[3],self.finalIfluxrerr[3][0],self.finalIfluxrerr[3][1]])
                    line.writerow(['Scaled STIS Flux',self.finalIfluxstis[0],self.finalIfluxster[0][0],self.finalIfluxster[0][1],self.finalIfluxstis[1],self.finalIfluxster[1][0],self.finalIfluxster[1][1],self.finalIfluxstis[2],self.finalIfluxster[2][0],self.finalIfluxster[2][1],self.finalIfluxstis[3],self.finalIfluxster[3][0],self.finalIfluxster[3][1]])
                    line.writerow(['','','']) #separator
                    line.writerow(['Fit Information','Lyα','OI 1302','OI 1305','OI 1306'])
                    line.writerow(['Fit Mode',self.finalFitmode[0],self.finalFitmode[1],self.finalFitmode[2],self.finalFitmode[3]])
                    line.writerow(['Airglow Mode',self.finalAirmode[0],self.finalAirmode[1],self.finalAirmode[2],self.finalAirmode[3]])
                    line.writerow(['Parameter Errors',self.finalEmethod[0],self.finalEmethod[1],self.finalEmethod[2],self.finalEmethod[3]])
                    line.writerow(['LP3 1327',['True' if ll else 'False' for ll in [self.specialLyA]][0],'N/A','N/A','N/A']) #only matters for LyA
                    line.writerow(['M Type',['True' if mm else 'False' for mm in [self.isM]][0],'N/A','N/A','N/A']) #only matters for LyA
                    line.writerow(['Side A Only',['True' if nn else 'False' for nn in [self.onlyA]][0],'-','-','-']) #applies for all 4 but redundant to include for each line
                    line.writerow(['','','']) #separator
                    line.writerow(['User Input','Lyα Input','OI 1302 Input','OI 1305 Input','OI 1306 Input'])
                    line.writerow(['Initial Guess Shift',self.finalUshifts[0],self.finalUshifts[1],self.finalUshifts[2],self.finalUshifts[3]])
                    line.writerow(['Initial Guess Scale',self.finalUscales[0],self.finalUscales[1],self.finalUscales[2],self.finalUscales[3]])
                    line.writerow(['Best Fit Shift',self.finalUfitshifts[0],self.finalUfitshifts[1],self.finalUfitshifts[2],self.finalUfitshifts[3]])
                    line.writerow(['Best Fit Scale',self.finalUfitscales[0],self.finalUfitscales[1],self.finalUfitscales[2],self.finalUfitscales[3]])
                    line.writerow(['Radial Velocity',self.finalRadial[0],self.finalRadial[1],self.finalRadial[2],self.finalRadial[3]])
                    line.writerow(['ISM Velocity',self.finalRadism[0],self.finalRadism[1],self.finalRadism[2],self.finalRadism[3]])
                    line.writerow(['Range Min',str(self.rangeLyA[0]),str(self.rangeOI2[0]),str(self.rangeOI5[0]),str(self.rangeOI6[0])])
                    line.writerow(['Range Max',str(self.rangeLyA[1]),str(self.rangeOI2[1]),str(self.rangeOI5[1]),str(self.rangeOI6[1])])
                    line.writerow(['N. Residual Cutoffs',self.finalCutoffs[0],self.finalCutoffs[1],self.finalCutoffs[2],self.finalCutoffs[3]])
                    line.writerow(['STIS Scale Factor',self.finalSTISscale[0],self.finalSTISscale[1],self.finalSTISscale[2],self.finalSTISscale[3]])
                    line.writerow(['','','']) #separator
                    line.writerow(['Bootstrap Information','Lyα','OI 1302','OI 1305','OI 1306'])
                    line.writerow(['Block Length',self.finalOptlen[0],self.finalOptlen[1],self.finalOptlen[2],self.finalOptlen[3]])
                    line.writerow(['Number of Blocks',self.finalNumblk[0],self.finalNumblk[1],self.finalNumblk[2],self.finalNumblk[3]])
                    line.writerow(['Number of Parameters',self.finalNumpar[0],self.finalNumpar[1],self.finalNumpar[2],self.finalNumpar[3]])
                    line.writerow(['Number of Samples',self.finalNumsmp[0],self.finalNumsmp[1],self.finalNumsmp[2],self.finalNumsmp[3]])
                    line.writerow(['Bootstrap Elapsed Time (s)',self.finalDeltaT[0],self.finalDeltaT[1],self.finalDeltaT[2],self.finalDeltaT[3]])

                bootLabel=['Lyα','OI2','OI5','OI6'] 
                for hh in range(0,len(self.bootDone)):
                    if self.bootDone[hh] and saveCheck.isChecked(): #only make these files if a bootstrap was performed and if the box was checked
                        oneBoot={}
                        oneBoot['BOOT']=self.finalRCBfit[hh]
                        oneBoot['STAR']=self.finalRCBste[hh]
                        oneBoot['PARS']=self.finalRCBpav[hh]
                        self.saveBoot(oneBoot,name2[0][:-4]+' Bootstrap for '+bootLabel[hh])
        
    def closeEvent(self,event):
        QtWidgets.QApplication.quit()
        #grants access back to the console
        
class missingWindow(QtWidgets.QDialog,missingUi):
    def __init__(self):
        super(missingWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal) #make this the only interactable window while open
        
        #star name lineEdit
        self.inputName.textChanged.connect(self.setStarname)
        
        #LP radio buttons
        self.life1.toggled.connect(self.position1)
        self.life2.toggled.connect(self.position2)
        self.life3.toggled.connect(self.position3)
        self.life4.toggled.connect(self.position4)
        self.life5.toggled.connect(self.position5)
        
        #special conditions checkboxes
        self.cen1327.stateChanged.connect(self.specialAirglow)
        self.mDwarf.stateChanged.connect(self.srSelector)
        self.justA.stateChanged.connect(self.plateA)
        
        #apply push button
        self.applyValues.clicked.connect(self.valApply)
        
        #placeholder values
        self.starName=None
        self.starLP=None
        self.starSpecial=False
        self.starRed=False
        self.starAonly=False
        
        #prevent user from closing this window
        self.canClose=False
        
        self.show()
        
    def setStarname(self):
        self.starName=self.inputName.text()
        
    def position1(self,selected):
        if selected:
            self.starLP=1
            
    def position2(self,selected):
        if selected:
            self.starLP=2
            
    def position3(self,selected):
        if selected:
            self.starLP=3
            
    def position4(self,selected):
        if selected:
            self.starLP=4
            
    def position5(self,selected):
        if selected:
            self.starLP=5
            
    def specialAirglow(self):
        if self.cen1327.isChecked():
            self.starSpecial=True #use LP3/1327 LyA template
        else:
            self.starSpecial=False #use general LyA template

    def srSelector(self):
        if self.mDwarf.isChecked():
            self.starRed=True #no SR component
        else:
            self.starRed=False #yes SR component

    def plateA(self):
        if self.justA.isChecked():
            self.starAonly=True #only use side A
        else:
            self.starAonly=False #use both side A and side B
            
    def valApply(self):
        if not isinstance(self.starLP,int):
            notEnough=QtWidgets.QMessageBox.warning(self,'Missing LP','Please select Lifetime Position!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notEnough==QtWidgets.QMessageBox.StandardButton.Ok:
                pass
        else:
            correctSettings=QtWidgets.QMessageBox.question(self,'Apply These Settings?','Is everything correct? Nothing can be changed unless you reload the COS file.',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
            if correctSettings==QtWidgets.QMessageBox.StandardButton.Yes:
                
                if not isinstance(self.starName,str):
                    GUI.starInput.setText('')
                    GUI.plotStarname=''
                else:
                    GUI.starInput.setText(self.starName)
                    GUI.plotStarname=self.starName
                    
                GUI.fileLP=self.starLP
                GUI.lifeLabel.setText('Lifetime Position: '+str(self.starLP))
                
                if self.starSpecial and self.starLP==3:
                    GUI.specialLyA=True
                    GUI.cen1327Label.setText('LP3 1327: True')
                else:
                    GUI.cen1327Label.setText('LP3 1327: False')
                        
                if self.starRed:
                    GUI.isM=True
                    GUI.spectypeLabel.setText('M Type: True')
                else:
                    GUI.spectypeLabel.setText('M Type: False')
                
                if self.starAonly:
                    GUI.onlyA=True
                    GUI.sideLabel.setText('Side A Only: True')
                else:
                    GUI.sideLabel.setText('Side A Only: False')

                self.canClose=True
                self.close()
            else:
                pass
            
    def closeEvent(self,event):
        if self.canClose:
            super(missingWindow,self).closeEvent(event)
        else:
            event.ignore()
            self.setWindowState(QtCore.Qt.WindowState.WindowMinimized)
            
class rangeWindow(QtWidgets.QDialog,rangeUi):
    def __init__(self):
        super(rangeWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal) #make this the only interactable window while open
        
        #initialize values in the user input boxes
        self.minLyA.setText(str(GUI.rangeLyA[0]))
        self.maxLyA.setText(str(GUI.rangeLyA[1]))
        self.minOI2.setText(str(GUI.rangeOI2[0]))
        self.maxOI2.setText(str(GUI.rangeOI2[1]))
        self.minOI5.setText(str(GUI.rangeOI5[0]))
        self.maxOI5.setText(str(GUI.rangeOI5[1]))
        self.minOI6.setText(str(GUI.rangeOI6[0]))
        self.maxOI6.setText(str(GUI.rangeOI6[1]))
        
        #lineEdit boxes for ranges
        self.minLyA.textChanged.connect(self.minimumLyA)
        self.maxLyA.textChanged.connect(self.maximumLyA)
        self.minOI2.textChanged.connect(self.minimumOI2)
        self.maxOI2.textChanged.connect(self.maximumOI2)
        self.minOI5.textChanged.connect(self.minimumOI5)
        self.maxOI5.textChanged.connect(self.maximumOI5)
        self.minOI6.textChanged.connect(self.minimumOI6)
        self.maxOI6.textChanged.connect(self.maximumOI6)
        
        #push buttons
        self.applyRange.clicked.connect(self.applyNew)
        self.cancelRange.clicked.connect(self.cancelNew)
        
        #temporary range placeholders 
        self.minLyAtemp=GUI.rangeLyA[0]
        self.maxLyAtemp=GUI.rangeLyA[1]
        self.minOI2temp=GUI.rangeOI2[0]
        self.maxOI2temp=GUI.rangeOI2[1]
        self.minOI5temp=GUI.rangeOI5[0]
        self.maxOI5temp=GUI.rangeOI5[1]
        self.minOI6temp=GUI.rangeOI6[0]
        self.maxOI6temp=GUI.rangeOI6[1]

        self.show()
        
    def minimumLyA(self):
        self.minLyAtemp=self.minLyA.text()
            
    def maximumLyA(self):
        self.maxLyAtemp=self.maxLyA.text()
            
    def minimumOI2(self):
        self.minOI2temp=self.minOI2.text()
            
    def maximumOI2(self):
        self.maxOI2temp=self.maxOI2.text()
            
    def minimumOI5(self):
        self.minOI5temp=self.minOI5.text()
            
    def maximumOI5(self):
        self.maxOI5temp=self.maxOI5.text()
            
    def minimumOI6(self):
        self.minOI6temp=self.minOI6.text()
            
    def maximumOI6(self):
        self.maxOI6temp=self.maxOI6.text()
            
    def applyNew(self):
        minmaxCheck=[self.minLyAtemp,self.maxLyAtemp,self.minOI2temp,self.maxOI2temp,self.minOI5temp,self.maxOI5temp,self.minOI6temp,self.maxOI6temp]
        allGood=True #assume everything looks good, then check if this is actually true
        
        #first, check if inputs can be interpreted properly, and if min<max is true for each line
        for i in range(0,len(minmaxCheck)):
            try:
                float(minmaxCheck[i])
            except:
                notFloat=QtWidgets.QMessageBox.warning(self,'Could Not Interpret Input(s)','One or more of the entered values\ncould not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
                if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                    allGood=False
                    break
            if i%2==1:
                if float(minmaxCheck[i])<=float(minmaxCheck[i-1]):
                    badRange=QtWidgets.QMessageBox.warning(self,'Invalid Range(s)','One or more minimum values are larger\nthan the corresponding maximum values!',QtWidgets.QMessageBox.StandardButton.Ok)
                    if badRange==QtWidgets.QMessageBox.StandardButton.Ok:
                        allGood=False
                        break
                    
        #then, check if the OI lines can be selected 
        lmOI2=((float(self.minOI2temp)<=GUI.waveCOS)&(GUI.waveCOS<=float(self.maxOI2temp))) #line mask
        fwOI2=1.0/GUI.errrCOS[lmOI2] #fit weights
        if np.any(fwOI2==np.inf):
            disableOI2=True #wait to disable the line until the end, after ensuring that everything is all good
            if GUI.currentLine==2:#only prevent the user from leaving this window if this is the currently selected line
                badOxygen=QtWidgets.QMessageBox.warning(self,'OI 1302 Contains Bad Flux Errors','The OI 1302 wavelength range contains flux error\nvalues that are zero, please edit the line range!',QtWidgets.QMessageBox.StandardButton.Ok)
                if badOxygen==QtWidgets.QMessageBox.StandardButton.Ok:
                    allGood=False
        else:
            disableOI2=False
        lmOI5=((float(self.minOI5temp)<=GUI.waveCOS)&(GUI.waveCOS<=float(self.maxOI5temp))) #line mask
        fwOI5=1.0/GUI.errrCOS[lmOI5] #fit weights
        if np.any(fwOI5==np.inf):
            disableOI5=True
            if GUI.currentLine==5:
                badOxygen=QtWidgets.QMessageBox.warning(self,'OI 1305 Contains Bad Flux Errors','The OI 1305 wavelength range contains flux error\nvalues that are zero, please edit the line range!',QtWidgets.QMessageBox.StandardButton.Ok)
                if badOxygen==QtWidgets.QMessageBox.StandardButton.Ok:
                    allGood=False
        else:
            disableOI5=False
        lmOI6=((float(self.minOI6temp)<=GUI.waveCOS)&(GUI.waveCOS<=float(self.maxOI6temp))) #line mask
        fwOI6=1.0/GUI.errrCOS[lmOI6] #fit weights
        if np.any(fwOI6==np.inf):
            disableOI6=True
            if GUI.currentLine==6:
                badOxygen=QtWidgets.QMessageBox.warning(self,'OI 1306 Contains Bad Flux Errors','The OI 1306 wavelength range contains flux error\nvalues that are zero, please edit the line range!',QtWidgets.QMessageBox.StandardButton.Ok)
                if badOxygen==QtWidgets.QMessageBox.StandardButton.Ok:
                    allGood=False  
        else:
            disableOI6=False
            
        #user input, make sure change is okay
        if GUI.fitIndex==0 and GUI.currentRange!=[float(self.minLyAtemp),float(self.maxLyAtemp)]:
            allGood=GUI.fitChanges()
            if not allGood:
                self.minLyA.setText(str(GUI.currentRange[0])) #return original values back to the input box
                self.maxLyA.setText(str(GUI.currentRange[1]))
        elif GUI.fitIndex==1 and GUI.currentRange!=[float(self.minOI2temp),float(self.maxOI2temp)]:
            allGood=GUI.fitChanges()
            if not allGood:
                self.minOI2.setText(str(GUI.currentRange[0])) #return original values back to the input box
                self.maxOI2.setText(str(GUI.currentRange[1]))
        elif GUI.fitIndex==2 and GUI.currentRange!=[float(self.minOI5temp),float(self.maxOI5temp)]:
            allGood=GUI.fitChanges()
            if not allGood:
                self.minOI5.setText(str(GUI.currentRange[0])) #return original values back to the input box
                self.maxOI5.setText(str(GUI.currentRange[1]))
        elif GUI.fitIndex==3 and GUI.currentRange!=[float(self.minOI6temp),float(self.maxOI6temp)]:
            allGood=GUI.fitChanges()
            if not allGood:
                self.minOI6.setText(str(GUI.currentRange[0])) #return original values back to the input box
                self.maxOI6.setText(str(GUI.currentRange[1]))
                        
        if allGood:
            #update range labels and lists to reflect new values
            GUI.rangeLyA=[float(self.minLyAtemp),float(self.maxLyAtemp)]
            GUI.labelLyA.setText('      Lyα Range: '+str(GUI.rangeLyA[0])+' - '+str(GUI.rangeLyA[1])+' Å')
            GUI.rangeOI2=[float(self.minOI2temp),float(self.maxOI2temp)]
            GUI.labelOI2.setText('OI 1302 Range: '+str(GUI.rangeOI2[0])+' - '+str(GUI.rangeOI2[1])+' Å') 
            GUI.rangeOI5=[float(self.minOI5temp),float(self.maxOI5temp)]
            GUI.labelOI5.setText('OI 1305 Range: '+str(GUI.rangeOI5[0])+' - '+str(GUI.rangeOI5[1])+' Å') 
            GUI.rangeOI6=[float(self.minOI6temp),float(self.maxOI6temp)]
            GUI.labelOI6.setText('OI 1306 Range: '+str(GUI.rangeOI6[0])+' - '+str(GUI.rangeOI6[1])+' Å') 
            
            #update the currently selected line range
            if GUI.fitIndex==0:
                GUI.currentRange=GUI.rangeLyA
            elif GUI.fitIndex==1:
                GUI.currentRange=GUI.rangeOI2  
            elif GUI.fitIndex==2:
                GUI.currentRange=GUI.rangeOI5  
            elif GUI.fitIndex==3:
                GUI.currentRange=GUI.rangeOI6  
                
            #if a fit currently exists, redo the fit with the new wavelength range, otherwise update plot range of currently selected line
            if GUI.fitExists:
                GUI.lineActions(GUI.currentRange,GUI.fitIndex)
            else:
                GUI.displayPlot(GUI.currentRange)
                
            #disable/enable any oxygen lines that require it
            if disableOI2:
                GUI.radioOI2.setDisabled(True)
                GUI.radioOI2.setToolTip('The current wavelength range for this line contains flux error values that are zero, try editing the line range')
            else:
                GUI.radioOI2.setDisabled(False)
                GUI.radioOI2.setToolTip('')
            if disableOI5:
                GUI.radioOI5.setDisabled(True)
                GUI.radioOI5.setToolTip('The current wavelength range for this line contains flux error values that are zero, try editing the line range')
            else:
                GUI.radioOI5.setDisabled(False)
                GUI.radioOI5.setToolTip('')
            if disableOI6:
                GUI.radioOI6.setDisabled(True)
                GUI.radioOI6.setToolTip('The current wavelength range for this line contains flux error values that are zero, try editing the line range')
            else:
                GUI.radioOI6.setDisabled(False)
                GUI.radioOI6.setToolTip('')        

            self.close()
            
    def cancelNew(self):
        self.close()

class stisWindow(QtWidgets.QMainWindow,stisUi):
    def __init__(self):
        super(stisWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal) #make this the only interactable window while open
        
        #matplotlib toolbar, wasn't supported in QDialog so I made this a QMainWindow
        self.addToolBar(NavToolbar(self.MplWidgetSTIS.canvas,self))
        
        #STIS slider stuff
        self.scalix=GUI.sfSTIS[GUI.fitIndex]
        self.sliderSTIS.setValue(int(self.scalix*(1000.0/10.0))) #slider goes from 0..1000, scaled to 0.0..10.0
        self.inputSTIS.setText(str(round(self.scalix,2))) #display current scale value
        self.inputSTIS.returnPressed.connect(self.changeSTIS) #pressing enter on lineEdit box applies user STIS scale
        self.sliderSTIS.valueChanged.connect(self.scaleSTIS) #as value is changed, scale is changed simultaneously
        
        #calculate the recovered spectrum using the current fit parameters
        if GUI.fitExists:
            if GUI.whichFit==1:
                self.tempRecov,self.tempError,self.tempRemov,self.tempRemer=GUI.recoverTrue(GUI.waveCOS,GUI.fluxCOS,GUI.errrCOS,GUI.pv[10]+GUI.allFitshifts[GUI.fitIndex],GUI.pv[11]*GUI.allFitscales[GUI.fitIndex],GUI.currentLine)
            else:
                self.tempRecov,self.tempError,self.tempRemov,self.tempRemer=GUI.oneSpectrum,GUI.oneError,GUI.oneRemove,GUI.oneRemerr
        
        #calculate and display the integrated fluxes (maybe I should do errorbars too?)
        if GUI.fitExists:
            self.intCOS=GUI.integrateFlux(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],self.tempRecov[GUI.allLinemasks[GUI.fitIndex]])
            self.ierCOS=GUI.coreEnforcer(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],self.tempRecov[GUI.allLinemasks[GUI.fitIndex]],self.tempError[GUI.allLinemasks[GUI.fitIndex]])
            self.cosIntvalue.setText('{:0.2e}'.format(self.intCOS)+' + '+'{:0.2e}'.format(self.ierCOS[0])+' - '+'{:0.2e}'.format(self.ierCOS[1]))
        self.maskSTIS=((GUI.currentRange[0]<=GUI.waveSTIS)&(GUI.waveSTIS<=GUI.currentRange[1]))
        self.intSTIS=GUI.integrateFlux(GUI.waveSTIS[self.maskSTIS],GUI.fluxSTIS[self.maskSTIS]*self.scalix)
        self.ierSTIS=GUI.coreEnforcer(GUI.waveSTIS[self.maskSTIS],GUI.fluxSTIS[self.maskSTIS]*self.scalix,GUI.errrSTIS[self.maskSTIS]*self.scalix)
        self.stisIntvalue.setText('{:0.2e}'.format(self.intSTIS)+' + '+'{:0.2e}'.format(self.ierSTIS[0])+' - '+'{:0.2e}'.format(self.ierSTIS[1]))
        
        #show the plot from the beginning, initial default is scale factor of 1.0
        self.firstSTIS=True
        self.plotSTIS()
        self.show()
        
    def changeSTIS(self): #changing the input box
        try:
            temp_val=int(round(float(self.inputSTIS.text())*(1000.0/10.0))) #convert value back to 0..1000 range
            isFloat=True
        except:
            notFloat=QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float!',QtWidgets.QMessageBox.StandardButton.Ok)
            if notFloat==QtWidgets.QMessageBox.StandardButton.Ok:
                isFloat=False
                
        if isFloat:
            if temp_val>1000:
                unscaled_val=1000
            elif temp_val<0:
                unscaled_val=0
            else:
                unscaled_val=temp_val
            self.scalix=unscaled_val*(10.0/1000.0)
            self.inputSTIS.setText(str(round(self.scalix,2)))
            self.sliderSTIS.setValue(unscaled_val)
            QtWidgets.QApplication.focusWidget().clearFocus() #after pressing enter, textbox is no longer selected
            self.intSTIS=GUI.integrateFlux(GUI.waveSTIS[self.maskSTIS],GUI.fluxSTIS[self.maskSTIS]*self.scalix)
            self.ierSTIS=GUI.coreEnforcer(GUI.waveSTIS[self.maskSTIS],GUI.fluxSTIS[self.maskSTIS]*self.scalix,GUI.errrSTIS[self.maskSTIS]*self.scalix)
            GUI.intSTISdat=self.intSTIS
            GUI.intSTISerr=self.ierSTIS #update GUI values
            GUI.sfSTIS[GUI.fitIndex]=self.scalix
            self.stisIntvalue.setText('{:0.2e}'.format(self.intSTIS)+' + '+'{:0.2e}'.format(self.ierSTIS[0])+' - '+'{:0.2e}'.format(self.ierSTIS[1]))
            
            self.plotSTIS()
    
    def scaleSTIS(self,value): #changing the slider itself
        self.scalix=value*(10.0/1000.0)
        self.inputSTIS.setText(str(round(self.scalix,2)))
        self.intSTIS=GUI.integrateFlux(GUI.waveSTIS[self.maskSTIS],GUI.fluxSTIS[self.maskSTIS]*self.scalix)
        self.ierSTIS=GUI.coreEnforcer(GUI.waveSTIS[self.maskSTIS],GUI.fluxSTIS[self.maskSTIS]*self.scalix,GUI.errrSTIS[self.maskSTIS]*self.scalix)
        GUI.intSTISdat=self.intSTIS
        GUI.intSTISerr=self.ierSTIS #update GUI values
        GUI.sfSTIS[GUI.fitIndex]=self.scalix
        self.stisIntvalue.setText('{:0.2e}'.format(self.intSTIS)+' + '+'{:0.2e}'.format(self.ierSTIS[0])+' - '+'{:0.2e}'.format(self.ierSTIS[1]))
        self.plotSTIS()
        
    def plotSTIS(self):
        if self.firstSTIS:
            self.stisXlim=GUI.currentXlim
            self.stisYlim=GUI.currentYlim 
            self.firstSTIS=False
        else:
            self.stisXlim=self.MplWidgetSTIS.canvas.axes.get_xlim()
            self.stisYlim=self.MplWidgetSTIS.canvas.axes.get_ylim()
        
        self.MplWidgetSTIS.canvas.axes.clear()
        self.MplWidgetSTIS.canvas.axes.set_title(GUI.plotStarname+' STIS Data')
        if GUI.fitExists:
            self.MplWidgetSTIS.canvas.axes.plot(GUI.waveCOS,self.tempRecov,color='C9',linewidth=2,label='Recovered COS Spectrum')
            self.MplWidgetSTIS.canvas.axes.fill_between(GUI.waveCOS,self.tempRecov+self.tempError,self.tempRecov-self.tempError,color='C9',alpha=0.4)
            self.MplWidgetSTIS.canvas.axes.set_title(GUI.plotStarname+' Comparison to STIS Data with Current Parameters')
        self.MplWidgetSTIS.canvas.axes.plot(GUI.waveSTIS,GUI.fluxSTIS*self.scalix,color='C2',linewidth=2,label='STIS Spectrum')
        self.MplWidgetSTIS.canvas.axes.fill_between(GUI.waveSTIS,(GUI.fluxSTIS+GUI.errrSTIS)*self.scalix,(GUI.fluxSTIS-GUI.errrSTIS)*self.scalix,color='C2',alpha=0.25)
        self.MplWidgetSTIS.canvas.axes.set_xlabel('Wavelength ($\AA$)')
        self.MplWidgetSTIS.canvas.axes.set_ylabel('Flux Density (erg $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)')
        self.MplWidgetSTIS.canvas.axes.set_xlim(self.stisXlim)
        self.MplWidgetSTIS.canvas.axes.set_ylim(self.stisYlim)
        self.MplWidgetSTIS.canvas.axes.legend()
        self.MplWidgetSTIS.canvas.draw()
        
class resultWindow(QtWidgets.QDialog,resultsUi):
    def __init__(self):
        super(resultWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal) #make this the only interactable window while open
        
        if not GUI.bootDone[GUI.fitIndex]:
            parErr=GUI.pe
        else:
            parErr=np.array(GUI.finalRCBpae[GUI.fitIndex]).transpose() 
        
        self.linecenValue.setText(str(round(GUI.pv[0],4))+' (fixed)')
        self.starvradValue.setText(str(round(GUI.pv[1],4))+self.prepErr(parErr[1]))
        self.gaussValue.setText(str(round(GUI.pv[2],4))+self.prepErr(parErr[2]))
        if GUI.fitIndex==0:
            self.lorentzValue.setText(str(round(GUI.pv[3],4))+self.prepErr(parErr[3]))
        self.ampValue.setText(str(round(GUI.pv[4],4))+self.prepErr(parErr[4]))
        if GUI.fitIndex==0:
            self.revwidthValue.setText(str(round(GUI.pv[5],4))+self.prepErr(parErr[5]))
            self.revdepthValue.setText(str(round(GUI.pv[6],4))+' (fixed)')
            self.coldensValue.setText(str(round(GUI.pv[7],4))+self.prepErr(parErr[7]))
            self.ismvradValue.setText(str(round(GUI.pv[8],4))+self.prepErr(parErr[8]))
            self.dopbValue.setText(str(round(GUI.pv[9],4))+' (fixed)')
        if GUI.whichAir==0:
            if GUI.whichFit==1:
                self.agshiftValue.setText(str(round(GUI.pv[10],4))+self.prepErr(parErr[10]))
                self.agscaleValue.setText(str(round(GUI.pv[11],4))+self.prepErr(parErr[11]))
            else: #display the original airglow errors, since this does not get bootstrapped
                self.agshiftValue.setText(str(round(GUI.pv[10],4))+self.prepErr(GUI.pe[10]))
                self.agscaleValue.setText(str(round(GUI.pv[11],4))+self.prepErr(GUI.pe[11]))      
        elif GUI.whichAir==1:
            self.agshiftValue.setText(str(round(GUI.pv[10],4))+' (fixed)')
            self.agscaleValue.setText(str(round(GUI.pv[11],4))+' (fixed)')         
        self.chiValue.setText(str(round(GUI.fitChi[0],4)))
        self.redchiValue.setText(str(round(GUI.fitChi[1],4)))
        self.bicValue.setText(str(round(GUI.fitBic,4)))
        
        if not GUI.bootDone[GUI.fitIndex]:
            if GUI.intStelerr[0]!=None:
                self.starfluxValue.setText('{:0.2e}'.format(GUI.intStellar)+' + '+'{:0.2e}'.format(GUI.intStelerr[0])+' - '+'{:0.2e}'.format(GUI.intStelerr[1]))
            else:
                self.starfluxValue.setText('{:0.2e}'.format(GUI.intStellar)+' (no error reported)')
        else:
            self.starfluxValue.setText('{:0.2e}'.format(GUI.intStellar)+' + '+'{:0.2e}'.format(GUI.finalRCBsie[GUI.fitIndex][0])+' - '+'{:0.2e}'.format(GUI.finalRCBsie[GUI.fitIndex][1])) 
        self.recovfluxValue.setText('{:0.2e}'.format(GUI.intRecover)+' + '+'{:0.2e}'.format(GUI.intRecverr[0])+' - '+'{:0.2e}'.format(GUI.intRecverr[1]))
        if GUI.readySTIS:
            self.stisfluxValue.setText('{:0.2e}'.format(GUI.intSTISdat)+' + '+'{:0.2e}'.format(GUI.intSTISerr[0])+' - '+'{:0.2e}'.format(GUI.intSTISerr[1])) #these are already scaled, do not need to multiply again
        
        self.show()
        
    def prepErr(self,err):
        #convert single float into a list if necessary
        if isinstance(err,float):
                err=[err] 
        #begin string formatting
        if len(err)==1:
            errstr=' +/- '
        elif len(err)==2:
            errstr=' + '
        #loop through error(s)
        for e in range(0,len(err)):
            try:
                errstr+=str(round(err[e],4))
            except:
                errstr+='N/A'
            if len(err)==2 and e!=1:
                errstr+=' - ' #add between errors
        return errstr
        
class residualWindow(QtWidgets.QMainWindow,residualsUi):
    def __init__(self):
        super(residualWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal) #make this the only interactable window while open
        
        #matplotlib toolbar, wasn't supported in QDialog so I made this a QMainWindow
        self.addToolBar(NavToolbar(self.MplWidgetResid.canvas,self))
        
        #apply cutoff when float value is inputted and user presses enter
        self.cutoffInput.returnPressed.connect(self.applyCutoff)
        
        #push buttons
        self.confirmCutoff.clicked.connect(self.saveCutoff)
        self.cancelCutoff.clicked.connect(self.closeCutoff)
        
        #calculate the normalized residuals based on the current best fit
        if GUI.whichFit==1:
            self.normres=((GUI.origBests[GUI.fitIndex]-GUI.fluxCOS[GUI.origMasks[GUI.fitIndex]])**2.0)/(GUI.errrCOS[GUI.origMasks[GUI.fitIndex]]**2.0)
        else:
            self.normres=((GUI.origBests[GUI.fitIndex]-GUI.oneSpectrum[GUI.origMasks[GUI.fitIndex]])**2.0)/(GUI.oneError[GUI.origMasks[GUI.fitIndex]]**2.0)
        
        self.tempCut=np.inf
        self.firstCutoff=True
        self.plotCutoff()
        self.show()
        
    def applyCutoff(self):
        try:
            self.tempCut=float(self.cutoffInput.text())
            self.plotCutoff()
            QtWidgets.QApplication.focusWidget().clearFocus()
        except:
            if self.cutoffInput.text()=='inf':
                self.tempCut=np.inf
                self.plotCutoff()
                QtWidgets.QApplication.focusWidget().clearFocus()
            else:
                QtWidgets.QMessageBox.warning(self,'Could not interpret input','The entered value could not be interpreted as a float or as infinity!',QtWidgets.QMessageBox.StandardButton.Ok)

    def plotCutoff(self):
        if self.firstCutoff:
            self.cutXlim=GUI.currentXlim
            self.cutYlim=GUI.currentYlim 
            self.firstCutoff=False
        else:
            self.cutXlim=self.MplWidgetResid.canvas.axes1.get_xlim()
            self.cutYlim=self.MplWidgetResid.canvas.axes1.get_ylim()
        
        self.MplWidgetResid.canvas.axes1.clear()
        self.MplWidgetResid.canvas.axes2.clear()
        self.MplWidgetResid.canvas.axes1.set_title(GUI.plotStarname+' Normalized Residual Analysis')
        if GUI.whichFit==1:
            self.MplWidgetResid.canvas.axes1.plot(GUI.waveCOS,GUI.fluxCOS,color='C0',linewidth=2,label='COS Data')
            self.MplWidgetResid.canvas.axes1.fill_between(GUI.waveCOS,GUI.fluxCOS+GUI.errrCOS,GUI.fluxCOS-GUI.errrCOS,color='C0',alpha=0.25)
        else:
            self.MplWidgetResid.canvas.axes1.plot(GUI.waveCOS,GUI.oneSpectrum,color='C9',linewidth=2,label='Recovered Spec.')
            self.MplWidgetResid.canvas.axes1.fill_between(GUI.waveCOS,GUI.oneSpectrum+GUI.oneError,GUI.oneSpectrum-GUI.oneError,color='C9',alpha=0.25)
        self.MplWidgetResid.canvas.axes1.plot(GUI.waveModel+GUI.allFitshifts[GUI.fitIndex],GUI.bestFit*GUI.allFitscales[GUI.fitIndex],color='C1',linewidth=2,label='Current Best Fit ('+str(GUI.allCutoffs[GUI.fitIndex])+')')
        try:
            self.MplWidgetResid.canvas.axes1.fill_between(GUI.waveModel+GUI.allFitshifts[GUI.fitIndex],(GUI.bestFit+GUI.intBesterr[0])*GUI.allFitscales[GUI.fitIndex],(GUI.bestFit-GUI.intBesterr[1])*GUI.allFitscales[GUI.fitIndex],color='C1',alpha=0.25)
        except:
            pass
        if GUI.allCutoffs[GUI.fitIndex]!=np.inf:
            self.MplWidgetResid.canvas.axes1.plot(GUI.waveCOS[GUI.origMasks[GUI.fitIndex]],GUI.origBests[GUI.fitIndex],color='C2',linewidth=2,linestyle='--',label='Original Best Fit (inf)')
            try:
                self.MplWidgetResid.canvas.axes1.fill_between(GUI.waveCOS[GUI.origMasks[GUI.fitIndex]]+GUI.allFitshifts[GUI.fitIndex],(GUI.origBests[GUI.fitIndex]+GUI.origBerrs[GUI.fitIndex][0])*GUI.allFitscales[GUI.fitIndex],(GUI.origBests[GUI.fitIndex]-GUI.origBerrs[GUI.fitIndex][1])*GUI.allFitscales[GUI.fitIndex],color='C2',alpha=0.25)
            except:
                pass
        self.MplWidgetResid.canvas.axes1.set_ylabel('Flux Density (erg $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)')
        self.MplWidgetResid.canvas.axes1.set_xlim(self.cutXlim)
        self.MplWidgetResid.canvas.axes1.set_ylim(self.cutYlim)
        self.MplWidgetResid.canvas.axes1.legend()
        self.MplWidgetResid.canvas.axes2.plot(GUI.waveCOS[GUI.origMasks[GUI.fitIndex]],self.normres,color='C7',linewidth=2,label='Original $\chi^2_R$')
        self.MplWidgetResid.canvas.axes2.axhline(self.tempCut,color='k',linewidth=2)
        self.MplWidgetResid.canvas.axes2.set_xlabel('Wavelength ($\AA$)')
        self.MplWidgetResid.canvas.axes2.set_ylabel('Normalized Residuals$^2$')
        self.MplWidgetResid.canvas.axes2.set_xlim(self.cutXlim)
        self.MplWidgetResid.canvas.axes2.legend()
        self.MplWidgetResid.canvas.draw()   

    def saveCutoff(self):
        if not GUI.useCutoff[GUI.fitIndex]: #first cutoff applied to the line
            GUI.useCutoff[GUI.fitIndex]=True
        GUI.allCutoffs[GUI.fitIndex]=self.tempCut
        GUI.allCutmasks[GUI.fitIndex]=(self.normres<=self.tempCut)
        GUI.lineActions(GUI.currentRange,GUI.fitIndex)
        self.close()

    def closeCutoff(self):
        self.close()      

class spectrumWindow(QtWidgets.QDialog,recoveredUi):
    def __init__(self):
        super(spectrumWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal) #make this the only interactable window while open
        
        #selection push buttons
        self.newButton.clicked.connect(self.createNew)
        self.cancelButton.clicked.connect(self.cancelRemoval)
        
        self.show()
        
    def createNew(self):
        GUI.removeMethod='New'
        self.close()
        
    def cancelRemoval(self):
        GUI.removeMethod='Cancel'
        self.close()                   
        
class bootstrapWindow(QtWidgets.QDialog,bootstrapUi):
    def __init__(self):
        super(bootstrapWindow,self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(path+'//3x3_icon.webp'))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal) #make this the only interactable window while open
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint) #always have this window on top, so plots don't get in the way
        
        #calculate values for both labels and RRCBB setup
        self.numpar=len(GUI.bestFit)-(GUI.fitChi[0]/GUI.fitChi[1]) #number of free parameters
        if GUI.whichFit==1:
            resids=GUI.fluxCOS[GUI.allLinemasks[GUI.fitIndex]]-GUI.bestFit
            self.modres=self.MWR(resids,GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]],len(GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]]),self.numpar)
        elif GUI.whichFit==2:
            resids=GUI.oneSpectrum[GUI.allLinemasks[GUI.fitIndex]]-GUI.bestFit
            self.modres=self.MWR(resids,GUI.oneError[GUI.allLinemasks[GUI.fitIndex]],len(GUI.oneError[GUI.allLinemasks[GUI.fitIndex]]),self.numpar)
        self.optlen,self.numblk=self.optimalLength(self.modres)
        self.numsmp=1000
        self.paramNumber.setText(str(int(round(self.numpar))))
        self.sampleNumber.setText(str(int(round(self.numsmp)))) #for now this is fixed, customize in the future
        self.lengthNumber.setText(str(int(round(self.optlen))))
        if self.blockOverride:
            self.lengthNumber.setText(str(int(round(self.optlen)))+'*') #inform the user about this forced block length
            self.lengthNumber.setToolTip('Optimal length makes too few blocks for generating 1000 unique samples, forcing block length to 5')
            self.blockNumber.setText(str(int(round(self.numblk)))+'*')
            self.blockNumber.setToolTip('Optimal length makes too few blocks for generating 1000 unique samples, forcing block length to 5')
        else:
            self.lengthNumber.setText(str(int(round(self.optlen))))
            self.lengthNumber.setToolTip('') #undo tool tip if no longer needed
            self.blockNumber.setText(str(int(round(self.numblk))))
            self.blockNumber.setToolTip('')
        self.bootRan=False
        
        #push buttons and progress bar
        if GUI.bootDone[GUI.fitIndex]: #if bootstrap already run, reconfigure buttons and load plotting data
            self.runButton.clicked.connect(self.plotBoot)
            self.runButton.setText('View RRCBB Plots')
            self.cancelButton.setText('Close')
            self.timeLabel.setText('Already Ran')
            self.bootTracker.setValue(1000) #full bar to represent it has already been run
            self.bootTracker.setFormat("%.1f%%" % 100.0)
            self.all_fit=GUI.finalRCBfit[GUI.fitIndex]
            self.all_ste=GUI.finalRCBste[GUI.fitIndex]
            self.intBesberr=GUI.finalRCBbfe[GUI.fitIndex]
            self.intBroferr=GUI.finalRCBsce[GUI.fitIndex]
            try:
                self.mdler=np.maximum(GUI.intBesterr[0],GUI.intBesterr[1])
            except:
                self.mdler=False
        else: #otherwise, load the normal button connections
            self.runButton.clicked.connect(self.runBoot)
            self.bootTracker.setFormat("%.1f%%" % 0.0)
        self.cancelButton.clicked.connect(self.closeBoot)
        
        
    def MWR(self,resid,error,numdp,numpr):
        #modified weighted residuals, the value that will be bootstrapped
        sqwgt=np.sqrt(1.0/error**2.0)
        meanr=np.sum(sqwgt*resid)/numdp
        mwres=np.sqrt(numdp/(numdp-numpr))*(sqwgt*resid-meanr)
        return mwres
    
    def optimalLength(self,tbr):
        est_optlen=int(np.ceil(optimal_block_length(tbr)[0].b_star_cb))
        est_numblk=int(np.ceil(len(tbr)/est_optlen))
        
        if np.ceil(len(tbr)/est_optlen)<5 or est_optlen==1:
            new_ratio=len(tbr)/4.0
            if new_ratio==np.floor(new_ratio):
                #inequality is Block Length < Data Length/4, so can't be equal to DL/4
                est_optlen=int(new_ratio-1.0)
            else:
                #if not an integer, floor it to get a block length that will create 5 blocks
                est_optlen=int(np.floor(new_ratio)) 
            est_numblk=5 #in the future I can let the user decide numbers (above minimum for # of unique samps)
            self.blockOverride=True
        else:
            self.blockOverride=False
            
        return est_optlen,est_numblk
    
    def multResample(self,arrays,block,numrep):
        #first, circle the final block around if needed
        origlen=len(arrays[0])
        if len(arrays[0])%block!=0:
            circle=block-len(arrays[0])%block
            for o in range(0,len(arrays)):
                arrays[o]=np.append(arrays[o],arrays[o][:circle])
        #now that all arrays are equally divisible by the block length, split them 
        for p in range(0,len(arrays)):
            arrays[p]=np.split(arrays[p],len(arrays[p])/block) #results in len(arrays[p])/block splits, with each split having length=block
        #next, we resample all arrays with replacement in the same way
        all_resamples=[0]*numrep #can't replace a numpy element with a numpy array apparently?
        for q in range(0,numrep):
            hatgrab=np.random.randint(0,len(arrays[0]),size=len(arrays[0]))
            holder=[0]*len(arrays)
            for r in range(0,len(arrays)):
                reswrep=np.array(arrays[r])[hatgrab].flatten() #resample w/replacement, may be inefficient according to interwebs
                #if the original length of the arrays is not divisible by the block length, the last bit of the array gets trimmed
                if origlen%block!=0:
                    reswrep=reswrep[:origlen]
                holder[r]=reswrep
            all_resamples[q]=holder
        
        return np.array(all_resamples)
    
    def errorProp(self,data_err,fit_err,rcb_data_err,rcb_fit_err,n_par):
        n_pts=len(data_err)
        r_bar_err2=(1.0/n_pts**2.0)*(n_pts+np.sum((fit_err**2.0)/(data_err**2.0)))
        mwr_err2=(n_pts/(n_pts-n_par))*(1.0+((rcb_fit_err**2.0)/(rcb_data_err**2.0))+r_bar_err2)
        y_boot_err2=fit_err**2.0+(mwr_err2*(data_err**2.0))
        y_boot_err=np.sqrt(y_boot_err2)
        return y_boot_err
    
    def plotBoot(self):
        fig,ax=plt.subplots(1,2,figsize=(16,9),layout='tight')
        fig.canvas.manager.set_window_title(GUI.plotStarname+'_'+GUI.lineLabel+'_Bootstrap')
        fig.supxlabel('Wavelength ($\AA$)',fontsize=18)
        plt.suptitle(GUI.plotStarname+' RRCBB Samples: '+GUI.lineLabel,fontsize=24)
        plt.sca(ax[0])
        if GUI.whichFit==1:
            plt.plot(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.fluxCOS[GUI.allLinemasks[GUI.fitIndex]],label='COS Data',linewidth=2,color='C0')
            plt.fill_between(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.fluxCOS[GUI.allLinemasks[GUI.fitIndex]]+GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.fluxCOS[GUI.allLinemasks[GUI.fitIndex]]-GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]],alpha=0.25,color='C0')
        else:
            plt.plot(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.trueSpectrum[GUI.allLinemasks[GUI.fitIndex]],label='Recovered Spec.',linewidth=2,color='C9')
            plt.fill_between(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.trueSpectrum[GUI.allLinemasks[GUI.fitIndex]]+GUI.propError[GUI.allLinemasks[GUI.fitIndex]],GUI.trueSpectrum[GUI.allLinemasks[GUI.fitIndex]]-GUI.propError[GUI.allLinemasks[GUI.fitIndex]],alpha=0.25,color='C9')
        plt.plot(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]]+GUI.allFitshifts[GUI.fitIndex],self.all_fit[0]*GUI.allFitscales[GUI.fitIndex],linewidth=0.025,color='k',label='RRCBB Samples')
        for n in range(1,len(self.all_fit)):
            plt.plot(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]]+GUI.allFitshifts[GUI.fitIndex],self.all_fit[n]*GUI.allFitscales[GUI.fitIndex],linewidth=0.025,color='k')
        plt.ylabel('Flux Density ($erg$ $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)',fontsize=18)
        OoM=ax[0].yaxis.get_offset_text()
        OoM.set_size(18)
        plt.tick_params(labelsize=18)
        leg=plt.legend(fontsize=15,loc=2)
        leg.get_lines()[1].set_linewidth(2) #change the linewidth or the RRCBB sample in the legend so it isnt nearly invisible
        plt.xticks(rotation=30)
        plt.sca(ax[1])
        plt.plot(GUI.allWaveinfs[GUI.fitIndex]+GUI.allFitshifts[GUI.fitIndex],GUI.selfreversalComp*GUI.allFitscales[GUI.fitIndex],label='Best Fit Stellar',linewidth=2,color='C6')
        try:
            plt.fill_between(GUI.allWaveinfs[GUI.fitIndex]+GUI.allFitshifts[GUI.fitIndex],(GUI.selfreversalComp+GUI.intProferr[0])*GUI.allFitscales[GUI.fitIndex],(GUI.selfreversalComp-GUI.intProferr[1])*GUI.allFitscales[GUI.fitIndex],color='C6',alpha=0.25)
        except:
            pass
        plt.plot(GUI.allWaveinfs[GUI.fitIndex]+GUI.allFitshifts[GUI.fitIndex],self.all_ste[0]*GUI.allFitscales[GUI.fitIndex],linewidth=0.025,color='k',label='RRCBB Samples')
        for n in range(1,len(self.all_ste)):
            plt.plot(GUI.allWaveinfs[GUI.fitIndex]+GUI.allFitshifts[GUI.fitIndex],self.all_ste[n]*GUI.allFitscales[GUI.fitIndex],linewidth=0.025,color='k')
        OoM=ax[1].yaxis.get_offset_text()
        OoM.set_size(18)
        plt.tick_params(labelsize=18)
        leg2=plt.legend(fontsize=15,loc=1)
        leg2.get_lines()[1].set_linewidth(2) #change the linewidth or the RRCBB sample in the legend so it isnt nearly invisible
        plt.xticks(rotation=30)
        plt.draw()
        
        fig,ax=plt.subplots(1,2,figsize=(16,9),layout='tight')
        fig.canvas.manager.set_window_title(GUI.plotStarname+'_'+GUI.lineLabel+'_Compare')
        fig.supxlabel('Wavelength ($\AA$)',fontsize=18)
        plt.suptitle(GUI.plotStarname+' Observed and Stellar Profiles: '+GUI.lineLabel,fontsize=24)
        plt.sca(ax[0])
        if GUI.whichFit==1:
            plt.plot(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.fluxCOS[GUI.allLinemasks[GUI.fitIndex]],label='COS Data',linewidth=2,color='C0')
            plt.fill_between(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.fluxCOS[GUI.allLinemasks[GUI.fitIndex]]+GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.fluxCOS[GUI.allLinemasks[GUI.fitIndex]]-GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]],alpha=0.65,color='C0')
        else:
            plt.plot(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.trueSpectrum[GUI.allLinemasks[GUI.fitIndex]],label='Recovered Spec.',linewidth=2,color='C9')
            plt.fill_between(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.trueSpectrum[GUI.allLinemasks[GUI.fitIndex]]+GUI.propError[GUI.allLinemasks[GUI.fitIndex]],GUI.trueSpectrum[GUI.allLinemasks[GUI.fitIndex]]-GUI.propError[GUI.allLinemasks[GUI.fitIndex]],alpha=0.65,color='C9')
        plt.plot(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]]+GUI.allFitshifts[GUI.fitIndex],GUI.bestFit*GUI.allFitscales[GUI.fitIndex],label='LMFIT Best Fit',linewidth=2,color='C1')
        if not isinstance(self.mdler,bool):
            plt.fill_between(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]]+GUI.allFitshifts[GUI.fitIndex],(GUI.bestFit+GUI.intBesterr[0])*GUI.allFitscales[GUI.fitIndex],(GUI.bestFit-GUI.intBesterr[1])*GUI.allFitscales[GUI.fitIndex],color='C1',alpha=0.45,label='LMFIT Error')
        plt.fill_between(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]]+GUI.allFitshifts[GUI.fitIndex],(GUI.bestFit+self.intBesberr[0])*GUI.allFitscales[GUI.fitIndex],(GUI.bestFit-self.intBesberr[1])*GUI.allFitscales[GUI.fitIndex],alpha=0.25,color='C2',label='RRCBB Error')
        plt.ylabel('Flux Density ($erg$ $cm^{-2}$ $s^{-1}$ $\AA^{-1}$)',fontsize=18)
        OoM=ax[0].yaxis.get_offset_text()
        OoM.set_size(18)
        plt.legend(fontsize=15)
        plt.tick_params(labelsize=18)
        plt.xticks(rotation=30)
        plt.sca(ax[1])
        plt.plot(GUI.allWaveinfs[GUI.fitIndex]+GUI.allFitshifts[GUI.fitIndex],GUI.selfreversalComp*GUI.allFitscales[GUI.fitIndex],linewidth=2,label='LMFIT Stellar',color='C6')
        try:
            plt.fill_between(GUI.allWaveinfs[GUI.fitIndex]+GUI.allFitshifts[GUI.fitIndex],(GUI.selfreversalComp+GUI.intProferr[0])*GUI.allFitscales[GUI.fitIndex],(GUI.selfreversalComp-GUI.intProferr[1])*GUI.allFitscales[GUI.fitIndex],color='C6',alpha=0.45,label='LMFIT Error')
        except:
            pass
        plt.fill_between(GUI.allWaveinfs[GUI.fitIndex]+GUI.allFitshifts[GUI.fitIndex],(GUI.selfreversalComp+self.intBroferr[0])*GUI.allFitscales[GUI.fitIndex],(GUI.selfreversalComp-self.intBroferr[1])*GUI.allFitscales[GUI.fitIndex],alpha=0.25,color='C2',label='RRCBB Error')
        OoM=ax[1].yaxis.get_offset_text()
        OoM.set_size(18)
        plt.legend(fontsize=15)
        plt.tick_params(labelsize=18)
        plt.xticks(rotation=30)
        plt.draw()
        
    def paramPercentile(self,pval,upper=84.135,lower=15.865):
        param_lists=[[] for _ in range(len(pval[0]))] #gather all of the sample params into a single list, for all params
        for j in range(0,len(pval)): #number of samples
            for k in range(0,len(pval[j])): #number of parameters
                param_lists[k].append(pval[j][k])
                
        top_v=[]
        mid_v=[]
        bot_v=[]
        for l in range(0,len(param_lists)): #get the percentiles for each parameter
            top,mid,bot=np.percentile(param_lists[l],[upper,50.0,lower],axis=0)
            top_v.append(top)
            mid_v.append(mid)
            bot_v.append(bot)
        upp_v=np.array(top_v)-np.array(mid_v)
        low_v=np.array(mid_v)-np.array(bot_v)
            
        return [upp_v,low_v],param_lists
    
    def profError(self):
        upf,mdf,lwf=np.percentile(self.all_fit,[84.135,50.0,15.865],axis=0) #this is here since the profiles were already created
        uuf=upf-mdf 
        llf=mdf-lwf
        return [uuf,llf]
    
    def runBoot(self):
        try:
            self.mdler=np.maximum(GUI.intBesterr[0],GUI.intBesterr[1]) #I should find a better way than this
        except:
            self.mdler=False
        if isinstance(self.mdler,bool):
            if GUI.whichFit==1:
                self.all_arrs=self.multResample([self.modres,GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]]],self.optlen,self.numsmp)
            elif GUI.whichFit==2:
                self.all_arrs=self.multResample([self.modres,GUI.oneError[GUI.allLinemasks[GUI.fitIndex]]],self.optlen,self.numsmp)
            all_LMF=np.array(1000*[[0]*len(GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]])])
            self.modlErr=np.array([0]*len(GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]]))
        else:
            if GUI.whichFit==1:
                self.all_arrs=self.multResample([self.modres,GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]],self.mdler],self.optlen,self.numsmp)
            elif GUI.whichFit==2:
                self.all_arrs=self.multResample([self.modres,GUI.oneError[GUI.allLinemasks[GUI.fitIndex]],self.mdler],self.optlen,self.numsmp)                
            all_LMF=self.all_arrs[:,2]
            self.modlErr=self.mdler
        if GUI.whichFit==1:
            all_RCB=GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]]*self.all_arrs[:,0]
        elif GUI.whichFit==2:
            all_RCB=GUI.oneError[GUI.allLinemasks[GUI.fitIndex]]*self.all_arrs[:,0]
        all_ERR=self.all_arrs[:,1]
            
        self.all_fit=[] #total fits 
        self.all_ste=[] #stellar fit components
        self.all_pav=[] #parameter values for each fit
        
        start=time.time()
        for j in range(0,len(all_RCB)):
            QtWidgets.QApplication.processEvents() #update progress bar frequently and without freezing
            if GUI.whichFit==1:
                booterror=self.errorProp(GUI.errrCOS[GUI.allLinemasks[GUI.fitIndex]],self.modlErr,all_ERR[j],all_LMF[j],self.numpar)
                Bbest,Bpara,Brept,Bchi2,Bbicc=GUI.totalModel(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.bestFit+all_RCB[j],booterror,GUI.lineCenters[GUI.fitIndex],GUI.finalRadial[GUI.fitIndex],GUI.finalRadism[GUI.fitIndex],GUI.finalUshifts[GUI.fitIndex],GUI.finalUscales[GUI.fitIndex],GUI.currentLine)
            elif GUI.whichFit==2:
                booterror=self.errorProp(GUI.oneError[GUI.allLinemasks[GUI.fitIndex]],self.modlErr,all_ERR[j],all_LMF[j],self.numpar)
                GUI.bootRunning=True
                Bbest,Bpara,Brept,Bchi2,Bbicc=GUI.totalModel(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],GUI.bestFit+all_RCB[j],booterror,GUI.lineCenters[GUI.fitIndex],GUI.finalRadial[GUI.fitIndex],GUI.finalRadism[GUI.fitIndex],0.0,0.0,GUI.currentLine)                
                GUI.bootRunning=False #only want this override to be in place for 2 part fitting
            Bpv,Bpe=GUI.extractParam(Bpara)
            Bstar,Brevr,Bisma,Bairg,Baire,Bconv,Bcopy=GUI.starComponents(GUI.waveCOS[GUI.allLinemasks[GUI.fitIndex]],Bpv[0],Bpv[1],Bpv[2],Bpv[3],Bpv[4],Bpv[5],Bpv[6],Bpv[7],Bpv[8],Bpv[9],Bpv[10],Bpv[11],GUI.currentLine)
            
            middl=time.time()
            estimt=(middl-start)/60.0 #minutes to get to this point
            fitvel=estimt/(j+1) #average minutes/fit
            remain=(1000-(j+1))*fitvel #minutes remaining
            self.bootTracker.setValue(j+1)
            self.bootTracker.setFormat("%.1f%%" % ((j+1)/10))
            self.timeLabel.setText(str(round(remain,2))+' Minutes')
            
            self.all_fit.append(Bbest)
            self.all_ste.append(Brevr)
            self.all_pav.append(Bpv)
            
        ended=time.time()
        self.delta=ended-start
        self.waitLabel.setText('Time to Complete:')
        self.timeLabel.setText(str(round(self.delta/60.0,2))+' Minutes')
        
        self.errp,self.allp=self.paramPercentile(self.all_pav)
        self.intBroferr,self.intBooterr=GUI.powerConstruct(GUI.allWaveinfs[GUI.fitIndex],self.allp[0][0],False,False,False,False,num_prf=self.numsmp,boot=self.allp)
        self.intBesberr=self.profError()
        
        self.runButton.clicked.disconnect() #change this button's functionality
        self.runButton.clicked.connect(self.plotBoot)
        self.runButton.setText('Display Plots')
        self.cancelButton.setText('Close')
        self.plotBoot()
        self.bootRan=True
    
    def closeBoot(self):
        if self.bootRan:
            GUI.finalOptlen[GUI.fitIndex]=self.optlen
            GUI.finalNumblk[GUI.fitIndex]=self.numblk
            GUI.finalNumpar[GUI.fitIndex]=self.numpar
            GUI.finalNumsmp[GUI.fitIndex]=self.numsmp
            GUI.finalDeltaT[GUI.fitIndex]=self.delta
            GUI.finalRCBfit[GUI.fitIndex]=self.all_fit
            GUI.finalRCBste[GUI.fitIndex]=self.all_ste
            GUI.finalRCBpav[GUI.fitIndex]=self.all_pav
            GUI.finalRCBpae[GUI.fitIndex]=self.errp
            GUI.finalRCBbfe[GUI.fitIndex]=self.intBesberr
            GUI.finalRCBsce[GUI.fitIndex]=self.intBroferr
            GUI.finalRCBsie[GUI.fitIndex]=self.intBooterr
            if not GUI.bootDone[GUI.fitIndex]:
                GUI.bootDone[GUI.fitIndex]=True
                GUI.finalEmethod[GUI.fitIndex]=True #True=RRCBB,False=LMFIT
        self.close()     

def starTemplate(waveGrid_s,lineCen_s,radialVel_s,fwhmG_s,fwhmL_s,fluxAmp_s,fwhmR_s,revAmpt_s,colDens_s,lineVel_s,lineDopb_s,airShift_s,airScale_s,airLine_s):
    #template for ISM  and self reversal attenuated voigt profile, for LyA
    strComp=GUI.stellarComponent(GUI.allWaveinfs[GUI.fitIndex],lineCen_s,radialVel_s,fwhmG_s,fwhmL_s,fluxAmp_s)
    if GUI.fitIndex==0: #LyA
        revComp=GUI.selfrevComponent(GUI.allWaveinfs[GUI.fitIndex],lineCen_s,radialVel_s,fwhmR_s,revAmpt_s)
        ismComp=GUI.tauComponent(GUI.allWaveinfs[GUI.fitIndex],colDens_s,lineVel_s,lineDopb_s)
        modFlux=(strComp*revComp)*ismComp
    else: #OI Triplet
        modFlux=strComp
    airComp,errComp=GUI.airglowComponent(waveGrid_s,airShift_s,airScale_s,airLine_s)
    
    if len(modFlux)<len(GUI.lsfCOS):
        diffLen=len(GUI.lsfCOS)-len(modFlux)
        GUI.lsfTrunc=GUI.lsfCOS[int(np.ceil(diffLen/2.0)):-int(np.ceil(diffLen/2.0))]
        tocFlux=np.convolve(modFlux,GUI.lsfTrunc,mode='same')
    else:
        tocFlux=np.convolve(modFlux,GUI.lsfCOS,mode='same')
        
    totalFlux=np.interp(waveGrid_s,GUI.allWaveinfs[GUI.fitIndex],tocFlux)+airComp
    #flux on the (masked) instrument wavegrid, convolved with infinite kernel then interpolated onto data grid
    
    return totalFlux

def run():
    global GUI #keep gui alive outside of the function
    app=QtWidgets.QApplication(sys.argv) 
    #qdarktheme.setup_theme('auto') #allows for following system light/dark mode
    if app is None: #keep kernel from dying
        app=QtWidgets.QApplication(sys.argv) 
    GUI=mainWindow()
    app.exec()
    
if __name__=='__main__':
    run()
