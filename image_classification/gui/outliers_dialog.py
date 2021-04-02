
from PyQt5.QtCore import *
from PyQt5.QtGui import *


import skimage.io
import os

from PyQt5.QtWidgets import *
from gui.outliers_dialog_ui import Ui_Dialog
from gui.outliers_groupBox import OutliersGroupBox

class Outliers(QDialog, Ui_Dialog):
    def __init__(self, dir_cropped_images, dir_images, sites_names, site_address, coordinates, rect):
        super().__init__()
        self.setupUi()

        self.directory_cropped_image = dir_cropped_images
        self.directory_images = dir_images
        self.sites = sites_names
        self.site_address = site_address # list of list containing each cluster's similar pics
        self.coordinates = coordinates
        self.rectangles = rect
        self.image_to_show = 0  # contains the number of image to show if the User presses >>> or <<<
        self.new_outliers_labels = [-1] * len(sites_names)

        #Contain original images
        self.scene = QGraphicsScene()
        self.image.setScene(self.scene)

        # Contains cropped image
        self.scene2 = QGraphicsScene()
        self.image2.setScene(self.scene2)
        self.item2 = QGraphicsPixmapItem()
        self.scene2.addItem(self.item2)

        if len(self.site_address[0]) > 3:
            self.site_address = [y[:2] for y in self.site_address]

        # Picture is changed -> reset user choises
        self.one_isPressed = False
        self.additional = None
        self.open_first_case = True

        self.indexes_to_iterate = list(range(len(self.site_address) - 1))
        self.indexes_to_iterate.append(-1)

        self.show_image()

    def showEvent(self, event):
        self.image.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.image2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)

    def setupUi(self):
        super().setupUi(self)

        self.forwardPushButton.clicked.connect(self.forward_picture)
        self.backwardPushButton.clicked.connect(self.back_picture)
        self.newClusterPushButton.clicked.connect(self.form_new_cluster)

    def form_new_cluster(self):

        self.new_outliers_labels[self.image_to_show] = max(self.indexes_to_iterate) + 1
        self.site_address.append([self.sites[self.image_to_show]])
        self.indexes_to_iterate.append(max(self.indexes_to_iterate) + 1)

        self.forward_picture()


    def clearWidget(self, layout):

         while layout.count():
             child = layout.takeAt(0)
             if child.widget():
                 child.widget().deleteLater()

    def forward_picture(self):

        if self.image_to_show == len(self.sites) - 1:
            self.image_to_show = len(self.sites) - 1
        else:
            self.image_to_show += 1

        self.show_image()

    def keyPressEvent(self, event):

        number_of_groups = (len(self.site_address)-1)
        print(number_of_groups)
        possible_buttons = [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5,
                            Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9][:number_of_groups]

        if event.key() in possible_buttons:
            cluster_value = event.key() - 48
            print(cluster_value)
            self.check_index = cluster_value
            self.check_checkboxes(1)
            self.forward_picture()

        elif event.key() == Qt.Key_A:
            self.back_picture()

        elif event.key() == Qt.Key_D:
            self.forward_picture()

    def back_picture(self):
        self.image_to_show -= 1
        if self.image_to_show < 0:
            self.image_to_show = 0

        self.show_image()


    def show_image(self):

        # check if outliers exist
        if len(self.sites) == 0:
            return

        # Show the current's outlier number
        self.outliersNumberLineEdit.setText("{}/{}".format(self.image_to_show,len(self.sites)-1))


        #Make sure that Checkboxes are empty and free to use.
        #Delete all widgets in the main layout
        self.clearWidget(self.verticalLayout_2)
        self.cluster_images = []

        # Show all clusters images -> delete all groupboxes and set it up again
        for i in self.indexes_to_iterate:

            if i != -1: # Not to pass outlier to outliers classification

                group_box = OutliersGroupBox(self.directory_cropped_image,
                                            self.site_address[self.indexes_to_iterate.index(i)], i)

                self.cluster_images.append(group_box)
                group_box.clusters_radiobutton.toggled.connect(self.check_checkboxes)
                self.verticalLayout_2.addWidget(group_box)


        if self.new_outliers_labels[self.image_to_show] != -1:

            # To mark already made choise

            self.cluster_images[self.new_outliers_labels[self.image_to_show]].clusters_radiobutton.blockSignals(True)
            self.cluster_images[self.new_outliers_labels[self.image_to_show]].clusters_radiobutton.setChecked(True)
            self.cluster_images[self.new_outliers_labels[self.image_to_show]].clusters_radiobutton.blockSignals(False)

        img_address2 = os.path.join(self.directory_cropped_image, "cropped_site{}.png".format(self.sites[self.image_to_show]))
        self.item2.setPixmap(QPixmap(img_address2))
        self.scene2.setSceneRect(self.item2.boundingRect())
        self.image2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)

        # Build path
        img_address = os.path.join(self.directory_images, 'site{}.jpg'.format(str(self.sites[self.image_to_show].split('_')[0])))
        self.scene.clear()
        self.item = QGraphicsPixmapItem()
        self.scene.addItem(self.item)
        self.item.setPixmap(QPixmap(img_address))

        # Image has got object -> Take each object and draw figures
        minr = int(self.rectangles.iloc[self.image_to_show, self.rectangles.columns.get_loc('Min row')])
        minc = int(self.rectangles.iloc[self.image_to_show, self.rectangles.columns.get_loc('Min column')])
        maxr = int(self.rectangles.iloc[self.image_to_show, self.rectangles.columns.get_loc('Max row')])
        maxc = int(self.rectangles.iloc[self.image_to_show, self.rectangles.columns.get_loc('Max column')])
        self.scene.setSceneRect(self.item.boundingRect())
        self.scene.addRect(QRectF(minc, minr, maxc - minc, maxr - minr), QPen(Qt.red, 5))
        self.image.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def check_checkboxes(self, state):

        sender = self.sender()
        if state:

            for i, cluster_image in enumerate(self.cluster_images):

                    if cluster_image.clusters_radiobutton == sender:
                        self.check_index = i
                        continue
                    else:
                        cluster_image.clusters_radiobutton.blockSignals(True)
                        cluster_image.clusters_radiobutton.setChecked(False)
                        cluster_image.clusters_radiobutton.blockSignals(False)

        self.new_outliers_labels[self.image_to_show] = self.check_index

    def get_newOutliersLabels(self):
        return self.new_outliers_labels


