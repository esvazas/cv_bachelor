import sys
import os

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pyqtgraph as pg

from gui.features_dialog_ui import Ui_Dialog

class FeaturesDialog(QDialog, Ui_Dialog):
    
    def __init__(self, featuresShapeNames, featureColorNames, featuresAdditional):
        super().__init__()
        self.featuresShapeNames = featuresShapeNames
        self.featureColorNames = featureColorNames
        self.featuresAdditional = featuresAdditional

        self.setupUi()

    def setupUi(self):
        super().setupUi(self)

        #Rankinis UI setup
        self.features_checkboxes = {}
        for name in self.featuresShapeNames:
            self.features_checkboxes[name] = QCheckBox(text=name)
            self.gridLayout.addWidget(self.features_checkboxes[name])

        for name in self.featureColorNames:
            self.features_checkboxes[name] = QCheckBox(text=name)
            self.gridLayout_3.addWidget(self.features_checkboxes[name])

        self.checkAll()
        self.selectAllPushButton.clicked.connect(self.checkAll)
        self.selectNonePushButton.clicked.connect(self.checkNone)


    def getCheckedFeatures(self):

        #add aditional required features for plotting
        return [key for key, item in self.features_checkboxes.items() if item.isChecked()] + self.featuresAdditional

    def checkAll(self):

        for name in self.features_checkboxes.keys():
            self.features_checkboxes[name].setChecked(True)

    def checkNone(self):

        for name in self.features_checkboxes.keys():
            self.features_checkboxes[name].setChecked(False)

