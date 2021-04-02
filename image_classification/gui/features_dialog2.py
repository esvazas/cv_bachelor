import sys
import os

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pyqtgraph as pg

from gui.features_dialog2_ui import Ui_Dialog

class FeaturesDialog2(QDialog, Ui_Dialog):
    
    def __init__(self, features, features2, checked_thumb, checked_eqhist, additional, pca_thumb, pca_eqhist, objectSize_thumb, objectSize_eqhist, max_thumb, max_eqhist):
        super().__init__()
        self.featuresNames = features #original thumb features
        self.featuresNames2 = features2
        self.checkedNames_thumb = checked_thumb #new and checked thumb features
        self.checkedNames_eqhist = checked_eqhist
        self.pca_thumb = pca_thumb
        self.pca_eqhist = pca_eqhist
        self.objectSize_thumb = objectSize_thumb
        self.objectSize_eqhist = objectSize_eqhist
        self.max_thumb = max_thumb
        self.max_eqhist = max_eqhist

        self.featuresAdditional = additional
        self.setupUi()

    def setupUi(self):
        super().setupUi(self)

        #Rankinis UI setup
        #Thumb images
        self.features_checkboxes = {}
        for name in self.featuresNames:
            self.features_checkboxes[name] = QCheckBox(text=name)

            if name in self.checkedNames_thumb:
                self.features_checkboxes[name].setChecked(True)

            name_index = list(self.featuresNames).index(name)

            if name_index < (len(self.featuresNames)/2):
                self.verticalLayout_6.addWidget(self.features_checkboxes[name])
            else:
                self.verticalLayout_7.addWidget(self.features_checkboxes[name])



        #Eq_hist images
        self.features_checkboxes2 = {}
        for name in self.featuresNames2:
            self.features_checkboxes2[name] = QCheckBox(text=name)

            if name in self.checkedNames_thumb:
                self.features_checkboxes[name].setChecked(True)

            #self.gridLayout_3.addWidget(self.features_checkboxes2[name])
            name_index = list(self.featuresNames2).index(name)

            if name_index < (len(self.featuresNames2)/2):
                self.verticalLayout_8.addWidget(self.features_checkboxes2[name])
            else:
                self.verticalLayout_9.addWidget(self.features_checkboxes2[name])

        self.PCA_checkBox.setChecked(self.pca_thumb)
        self.PCA_checkBox2.setChecked(self.pca_eqhist)

        self.doubleSpinBox_2.setValue(self.objectSize_thumb)
        self.doubleSpinBox.setValue(self.objectSize_eqhist)

        self.doubleSpinBox_2.setMinimum(self.objectSize_thumb)
        self.doubleSpinBox.setMinimum(self.objectSize_eqhist)

        self.doubleSpinBox_2.setMaximum(self.max_thumb)
        self.doubleSpinBox.setMaximum(self.max_eqhist)

        self.selectAllPushButton_2.clicked.connect(self.checkAll)
        self.selectNonePushButton_2.clicked.connect(self.checkNone)
        self.selectAllPushButton2.clicked.connect(self.checkAll2)
        self.selectNonePushButton2.clicked.connect(self.checkNone2)

    def getObjectsSize_thumb(self):
        return self.doubleSpinBox_2.value()

    def getObjectsSize_eqhist(self):
        return self.doubleSpinBox.value()

    def getCheckedFeatures(self):

        #add aditional required features for plotting
        return [key for key, item in self.features_checkboxes.items() if item.isChecked()] + self.featuresAdditional
        #return ['Blue part', 'Green part','Red part','Shapes Area','Entropy std','Entropy mean'] + self.featuresAdditional

    def getCheckedFeatures2(self):

        #add aditional required features for plotting
        return [key for key, item in self.features_checkboxes2.items() if item.isChecked()] + self.featuresAdditional
        #return ['Blue part', 'Green part','Red part','Shapes Area','Entropy std','Entropy mean'] + self.featuresAdditional

    def getCheckedPCA(self):
        return self.PCA_checkBox.isChecked()

    def getCheckedPCA2(self):
        return self.PCA_checkBox2.isChecked()

    def checkAll(self):

        for name in self.features_checkboxes.keys():
            self.features_checkboxes[name].setChecked(True)

    def checkNone(self):

        for name in self.features_checkboxes.keys():
            self.features_checkboxes[name].setChecked(False)

    def checkAll2(self):

        for name in self.features_checkboxes2.keys():
            self.features_checkboxes2[name].setChecked(True)

    def checkNone2(self):

        for name in self.features_checkboxes2.keys():
            self.features_checkboxes2[name].setChecked(False)



