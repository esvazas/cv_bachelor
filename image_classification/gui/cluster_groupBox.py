import os

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PyQt5 import QtCore


class ClusterGroupBox(QGroupBox):

    def __init__(self, directory_cropped_image, site_address, label, minimum, maximum, label_freq):
        super().__init__()

        self.images_qlabels = {} # contain cluster images
        self.images_qHorizontalLayout = None # contain a layout which will be filled with spinboxes and labels
        self.clusters_spinboxes = None
        self.clusters_textLabels = None #labels next to the spinboxes

        self.directory_cropped_image = directory_cropped_image
        self.site_address = site_address #single cluster sites
        self.label = label #single label
        self.minimum = minimum # to set up possible choises for the spinbox
        self.maximum = maximum
        self.label_freq = label_freq

        QGroupBox.setMaximumWidth(self, 150)

        self.setup_ClusterImages()

    def setup_ClusterImages(self):

        # To the existing Group Box add Vertical layout
        self.images_qVerticalLayout = QtWidgets.QVBoxLayout()

        # For Vertical Layout add the multiple pictures (can be just one!)
        for name in self.site_address:
            self.images_qlabels[name] = QLabel(text=name)
            self.images_qlabels[name].setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.images_qlabels[name].setMaximumSize(140, 140)
            self.images_qVerticalLayout.addWidget(self.images_qlabels[name])
            img = os.path.join(self.directory_cropped_image, "cropped_site{}.png".format(name))
            pixmap = QPixmap(img)
            self.images_qlabels[name].setPixmap(pixmap.scaled(self.images_qlabels[name].width(),
                                                                self.images_qlabels[name].height(),
                                                                Qt.KeepAspectRatio))

        # Add Horizontal Layout
        self.images_qHorizontalLayout = QtWidgets.QHBoxLayout()
        self.images_qVerticalLayout.addLayout(self.images_qHorizontalLayout)

        # Fill the last H Layout with text labels and spinboxes
        self.clusters_textLabels = QtWidgets.QLabel(text="{}: #{}".format(str(self.label), self.label_freq))
        self.clusters_textLabels.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.images_qHorizontalLayout.addWidget(self.clusters_textLabels)

        self.clusters_spinboxes = QtWidgets.QSpinBox()
        self.clusters_spinboxes.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.clusters_spinboxes.setMinimum(self.minimum)
        self.clusters_spinboxes.setMaximum(self.maximum)
        self.clusters_spinboxes.setValue(self.label)
        self.images_qHorizontalLayout.addWidget(self.clusters_spinboxes)

        self.setLayout(self.images_qVerticalLayout)

    def getSpinBox(self):
        return self.clusters_spinboxes.value()
