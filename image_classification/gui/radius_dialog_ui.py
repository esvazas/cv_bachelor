# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'radius_dialog_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(274, 294)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radiusDoubleSpinBox = QtWidgets.QDoubleSpinBox(Dialog)
        self.radiusDoubleSpinBox.setDecimals(0)
        self.radiusDoubleSpinBox.setMaximum(10000.0)
        self.radiusDoubleSpinBox.setObjectName("radiusDoubleSpinBox")
        self.horizontalLayout.addWidget(self.radiusDoubleSpinBox)
        self.radiusHorizontalSlider = QtWidgets.QSlider(Dialog)
        self.radiusHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.radiusHorizontalSlider.setObjectName("radiusHorizontalSlider")
        self.horizontalLayout.addWidget(self.radiusHorizontalSlider)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))

