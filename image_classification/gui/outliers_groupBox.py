from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import os


class OutliersGroupBox(QGroupBox):

    def __init__(self, directory_cropped_image, site_address, label):
        super().__init__()

        self.images_qlabels = {} # contain cluster images
        self.images_qHorizontalLayout = None # contain a layout which will be filled with spinboxes and labels


        self.directory_cropped_image = directory_cropped_image
        self.site_address = site_address #single cluster sites
        self.label = label #single label

        self.setupSignals()

    def setupSignals(self):

        # To the existing Group Box add Vertical layout
        self.images_qHorizontalLayout = QtWidgets.QHBoxLayout()

        # Fill the last H Layout with text labels and spinboxes
        self.clusters_radiobutton = QtWidgets.QRadioButton(text=str(self.label))
        self.clusters_radiobutton.setChecked(False)
        self.images_qHorizontalLayout.addWidget(self.clusters_radiobutton)

        # For Vertical Layout add the multiple pictures (can be just one!)
        for name in self.site_address:
            self.images_qlabels[name] = QLabel(text=name)
            self.images_qlabels[name].setMaximumSize(100, 100)
            self.images_qHorizontalLayout.addWidget(self.images_qlabels[name])
            img = os.path.join(self.directory_cropped_image,("cropped_site{}.png".format(name)))
            pixmap = QPixmap(img)
            self.images_qlabels[name].setPixmap(pixmap.scaled(self.images_qlabels[name].width(),
                                                                self.images_qlabels[name].height(),
                                                                Qt.KeepAspectRatio))

        self.setLayout(self.images_qHorizontalLayout)




