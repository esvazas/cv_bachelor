import sys
import os, re
import warnings
import errno
import numpy as np
import pandas as pd
import multiprocessing
from copy import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FingureCanvas
from matplotlib.figure import Figure
import skimage.io
import skimage


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore


from logic import identification
import gui.set_tips_and_buttons as set_tips_and_buttons
from gui.mainwindow_ui import Ui_MainWindow
from gui.features_dialog import FeaturesDialog

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #Contain original images
        self.scene = QGraphicsScene()
        self.image.setScene(self.scene)
        self.item = QGraphicsPixmapItem()
        self.scene.addItem(self.item)

        self.scene2 = QGraphicsScene()
        self.image2.setScene(self.scene2)
        self.item2 = QGraphicsPixmapItem()
        self.scene2.addItem(self.item2)

        #Contain images after processing
        self.scene3 = QGraphicsScene()
        self.image3.setScene(self.scene3)
        self.scene4 = QGraphicsScene()
        self.image4.setScene(self.scene4)

        self.featuresShapeNames = ['Shapes Area', 'Shapes Perimeter', 'Centroids row', 'Centroids column', 'Eccentricity',
                                   'Orientation', 'Area/Square diff', 'Area/Circle diff',
                                   'Object Perimeter/Circle Perimeter', 'Entropy', 'Sobel']
        self.featuresColorNames = ['Red part', 'Green part', 'Blue part', 'Gray', 'Color inertia', 'Texture', 'Moments']
        self.featuresAdditional = ['RGB clusters number', 'Masked_image', 'Oval', 'Rect', 'Bounding']
        self.featuresToRun = self.featuresShapeNames + self.featuresColorNames + self.featuresAdditional

        #  Background image -> used in identification.process image.
        #  Used for making convex hull around background points.
        self.background_img = None
        self.all_objects_thumb = None
        self.all_objects_eq_hist = None

        set_tips_and_buttons.setTips(self)
        self.useOriginalCheckBox.setChecked(True)
        self.useContrastCheckBox.setChecked(False)

        self.objectsFeaturesThread = ObjectsFeaturesThread()
        self.objectsFeaturesThread2 = ObjectsFeaturesThread2()

        self.bar = self.menuBar()
        self.Files = self.bar.addMenu('Files')

        self.afterDetectionStatAct = QAction('Pictures folder', self, checkable=True)
        self.afterDetectionStatAct.setChecked(False)
        self.afterDetectionStatAct.triggered.connect(self.openFileNamesDialog)
        self.Files.addAction(self.afterDetectionStatAct)

        self.afterDetectionStatAct2 = QAction('Pictures folder 2', self, checkable=True)
        self.afterDetectionStatAct2.setChecked(False)
        self.afterDetectionStatAct2.triggered.connect(self.openFileNamesDialog)
        self.Files.addAction(self.afterDetectionStatAct2)

        self.MDStatAct = QAction('MD file', self, checkable=True)
        self.MDStatAct.setChecked(False)
        self.MDStatAct.triggered.connect(self.openFileNamesDialog)
        self.Files.addAction(self.MDStatAct)

        self.objectsStatAct = QAction('Objects folder', self, checkable=True)
        self.objectsStatAct.setChecked(False)
        self.objectsStatAct.triggered.connect(self.openFileNamesDialog)
        self.Files.addAction(self.objectsStatAct)

        self.featuresMenuBar = self.bar.addMenu('Features')
        self.featuresRunStatAct = QAction("Select features", self)
        self.featuresRunStatAct.triggered.connect(self.openFeaturesDialog)
        self.featuresMenuBar.addAction(self.featuresRunStatAct)

        self.oppositeMenuBar = self.bar.addMenu('Opposite Objects')
        self.oppositeRunStatAct = QAction("Generate", self)
        self.oppositeRunStatAct.triggered.connect(self.show_figures)
        self.oppositeMenuBar.addAction(self.oppositeRunStatAct)

        self.cpuCoresSpinBox.setMinimum(1)
        self.cpuCoresSpinBox.setMaximum(multiprocessing.cpu_count())
        self.cpuCoresSpinBox.setValue(multiprocessing.cpu_count())
        self.cpuCoresSpinBox.setEnabled(True)
        self.selectAllCpuCorescheckBox.setChecked(True)

        self.setupSignals()

    def setupSignals(self):
        self.setGeometry(10, 10, 640, 480)
        self.show()

        self.stopButton.clicked.connect(QApplication.instance().quit)
        self.runInitializationButton.clicked.connect(self.run_identification)
        self.runObjectsButton.clicked.connect(self.run_all_objects)

        self.imageNumHorizontalSlider.valueChanged.connect(self.imageNumSpinbox.setValue)
        self.imageNumSpinbox.valueChanged.connect(self.imageNumHorizontalSlider.setValue)
        self.imageNumSpinbox.valueChanged.connect(self.show_image)
        self.cpuCoresSpinBox.valueChanged.connect(self.cpuCores_changed)
        self.selectAllCpuCorescheckBox.stateChanged.connect(self.cpuCores_changed)

        self.channelComboBox.currentTextChanged.connect(self.filter_changed)
        self.channelComboBox_2.currentTextChanged.connect(self.filter_changed)
        self.noiseComboBox.currentTextChanged.connect(self.filter_changed)
        self.noiseComboBox_2.currentTextChanged.connect(self.filter_changed)
        self.edgeComboBox.currentTextChanged.connect(self.filter_changed)
        self.edgeComboBox_2.currentTextChanged.connect(self.filter_changed)
        self.noise2ComboBox.currentTextChanged.connect(self.filter_changed)
        self.noise2ComboBox_2.currentTextChanged.connect(self.filter_changed)
        self.thresholdingComboBox.currentTextChanged.connect(self.filter_changed)
        self.thresholdingComboBox_2.currentTextChanged.connect(self.filter_changed)
        self.binClosingComboBox.currentTextChanged.connect(self.filter_changed)
        self.binClosingComboBox_2.currentTextChanged.connect(self.filter_changed)

    def show_figures(self):

        coordinates_thumb = self.all_objects_thumb[['site nr',  'Min row', 'Min column', 'Max row', 'Max column']]
        #coordinates_eqhist = self.all_objects_eq_hist[['site nr',  'Min row', 'Min column', 'Max row', 'Max column']]

        path_save_thumb = os.path.join(os.path.split(self.path_read_thumb2)[0], 'Opposite thumb')
        #path_save_eqhist = os.path.join(os.path.split(self.path_read_eq_hist2)[0], 'Opposite eq_hist')

        try:  # making the directory
            os.makedirs(path_save_thumb)
            #os.makedirs(path_save_eqhist)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.statusbar.showMessage("Generating Original pictures")
        for img_nr in self.sites_thumb2:

            img = skimage.io.imread(os.path.join(self.path_read_thumb2, 'site{}.jpg'.format(img_nr)))
            rows = coordinates_thumb.loc[coordinates_thumb['site nr'] == img_nr, :]

            fig, ax = plt.subplots(1)
            ax.imshow(img)

            if rows.shape[0] >0:
                for small_nr in list(rows.index):
                # Image has got object -> Take each object and draw figures
                    minr = coordinates_thumb.loc[small_nr, 'Min row']
                    minc = coordinates_thumb.loc[small_nr, 'Min column']
                    maxr = coordinates_thumb.loc[small_nr, 'Max row']
                    maxc = coordinates_thumb.loc[small_nr, 'Max column']

                    # Create a Rectangle patch
                    rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                        fill=False, edgecolor='salmon', linewidth=4)
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                ax.set_axis_off()
                fig.savefig(os.path.join(path_save_thumb, 'site{}.jpg'.format(img_nr)), bbox_inches='tight', pad_inches=0)
            else:
                # Image dont have objects -> Issaugoti pati paveiksla
                 ax.set_axis_off()
                 fig.savefig(os.path.join(path_save_thumb, 'site{}.jpg'.format(img_nr)), bbox_inches='tight', pad_inches=0)

        self.statusbar.clearMessage()
        # self.statusbar.showMessage("Generating Equalized Histogram pictures")
        #
        # for img_nr in self.sites_eq_hist2:
        #
        #     img = skimage.io.imread(os.path.join(self.path_read_eq_hist2, 'site{}.jpg'.format(img_nr)))
        #     rows = coordinates_eqhist.loc[coordinates_eqhist['site nr'] == img_nr, :]
        #
        #     fig, ax = plt.subplots(1)
        #     ax.imshow(img)
        #
        #     if rows.shape[0] > 0:
        #         for small_nr in list(rows.index):
        #             # Image has got object -> Take each object and draw figures
        #             minr = coordinates_eqhist.loc[small_nr, 'Min row']
        #             minc = coordinates_eqhist.loc[small_nr, 'Min column']
        #             maxr = coordinates_eqhist.loc[small_nr, 'Max row']
        #             maxc = coordinates_eqhist.loc[small_nr, 'Max column']
        #
        #             # Create a Rectangle patch
        #             rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                                                 fill=False, edgecolor='salmon', linewidth=4)
        #             # Add the patch to the Axes
        #             ax.add_patch(rect)
        #         ax.set_axis_off()
        #         fig.savefig(os.path.join(path_save_eqhist, 'site{}.jpg'.format(img_nr)), bbox_inches='tight',
        #                     pad_inches=0)
        #     else:
        #         # Image dont have objects -> Issaugoti pati paveiksla
        #         ax.set_axis_off()
        #         fig.savefig(os.path.join(path_save_eqhist, 'site{}.jpg'.format(img_nr)), bbox_inches='tight',
        #                     pad_inches=0)
        #
        # self.statusbar.clearMessage()

    def fitAllInView(self):
        self.image.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.image2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)
        self.image3.fitInView(self.scene3.sceneRect(), Qt.KeepAspectRatio)
        self.image4.fitInView(self.scene4.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super(MainWindow, self).resizeEvent(event)
        self.fitAllInView()

    def filter_changed(self):
        set_tips_and_buttons.filters_changed(self)

    def cpuCores_changed(self):

        sender = self.sender()

        if sender.objectName() == 'cpuCoresSpinBox':
            if self.cpuCoresSpinBox.value() == multiprocessing.cpu_count():
                self.selectAllCpuCorescheckBox.setChecked(True)
            else:
                self.selectAllCpuCorescheckBox.setChecked(False)

        elif sender.objectName() == 'selectAllCpuCorescheckBox':
            if self.selectAllCpuCorescheckBox.isChecked():
                self.cpuCoresSpinBox.setValue(multiprocessing.cpu_count())


    def openFeaturesDialog(self):
        dialog = FeaturesDialog(self.featuresShapeNames, self.featuresColorNames, self.featuresAdditional)
        if dialog.exec():
            # Ok
            self.featuresToRun = (dialog.getCheckedFeatures())
        else:
            # Cancel
            dialog.close()

    def run_identification(self):

        # Check if addresses are correct and already given
        if (('background_img' == None) or
            ('path_read_thumb' not in dir(self)) or
            ('path_save_pics_thumb' not in dir(self)) or
            ('path_read_eq_hist' not in dir(self)) or
            ('path_save_pics_eq_hist' not in dir(self))):

            QMessageBox.information(self, 'No required files',
                                    "Please select Objects folder and MD file beforehand.")
            return

        warnings.filterwarnings("ignore")
        self.statusbar.showMessage('Processing photos...')

        if (str(self.edgeComboBox.currentText()) == 'Canny'):
            self.noiseComboBox.setCurrentText('None')

        self.file_read_thumb = os.path.join(self.path_read_thumb, "site{}.jpg".format(str(self.imageNumSpinbox.value())))
        self.file_read_eq_hist = os.path.join(self.path_read_eq_hist, "site{}.jpg".format(str(self.imageNumSpinbox.value())))
        warnings.filterwarnings("ignore")

        all_o_D_2, self.img2, self.patches2, site_nr2 = identification.get_regions_from_image(
                    self.file_read_eq_hist, self.path_save_pics_eq_hist,
                    {'features': self.featuresToRun, 'plot_feature': False},
                    channel=self.channelComboBox_2.currentText().lower(),
                    noise_filter=self.noiseComboBox_2.currentText().lower(),
                    edge_detection=self.edgeComboBox_2.currentText().lower(),
                    noise_filter2=self.noise2ComboBox_2.currentText().lower(),
                    thresholding=self.thresholdingComboBox_2.currentText().lower(),
                    closing=self.binClosingComboBox_2.currentText().lower(),
                    fill_holes=self.fillingHolesComboBox_2.currentText().lower(),
                    filter_params=[None, (self.noiseSpinbox_2.value()),
                          (self.edgeSpinbox_2.value()),
                          (self.noise2Spinbox_2.value()),
                          (self.bin_closingSpinbox_2.value()),
                          (self.thresholding_minSpinbox_2.value()), self.background_img, 1],
                    plot_filters=self.plotFiltersCheckBox_3.isChecked(),
                    plot_object=self.plotObjectsCheckBox_3.isChecked(),
                    min_region_size=(self.minRegionSpinbox_2.value()), cropped_image_save=False)

        canvas2 = Canvas(self, self.img2, self.patches2, self.scene2.width()/100, self.scene2.height()/100)
        self.scene3.addWidget(canvas2)
        self.image3.setScene(self.scene3)
        self.statusbar.clearMessage()

        all_o_D_2, self.img1, self.patches1, site_nr1 = identification.get_regions_from_image(
            self.file_read_thumb, self.path_save_pics_thumb,
            {'features': self.featuresToRun, 'plot_feature': False},
            channel=self.channelComboBox.currentText().lower(),
            noise_filter=self.noiseComboBox.currentText().lower(),
            edge_detection=self.edgeComboBox.currentText().lower(),
            noise_filter2=self.noise2ComboBox.currentText().lower(),
            thresholding=self.thresholdingComboBox.currentText().lower(),
            closing=self.binClosingComboBox.currentText().lower(),
            fill_holes=self.fillingHolesComboBox.currentText().lower(),
            filter_params=[None, (self.noiseSpinbox.value()),
                           (self.edgeSpinbox.value()),
                           (self.noise2Spinbox.value()),
                           (self.bin_closingSpinbox.value()),
                           (self.thresholding_minSpinbox.value()), self.background_img, 0],
            plot_filters=self.plotFiltersCheckBox.isChecked(),
            plot_object=self.plotObjectsCheckBox.isChecked(),
            min_region_size=(self.minRegionSpinbox.value()), cropped_image_save=False)

        self.canvas1 = Canvas(self, self.img1, self.patches1, self.scene2.width()/100, self.scene2.height()/100)
        self.scene4.addWidget(self.canvas1)
        self.image4.setScene(self.scene4)
        self.statusbar.clearMessage()

    def run_all_objects(self, checked, multiprocess=True):

        # Check if addresses are correct and already given
        if (('background_img' == None) or
                ('path_read_thumb' not in dir(self)) or
                ('path_save_pics_thumb' not in dir(self)) or
                ('path_read_eq_hist' not in dir(self)) or
                ('path_save_pics_eq_hist' not in dir(self))):
            QMessageBox.information(self, 'No required files',
                                    "Please select Objects folder and MD file beforehand.")
            return
        try:  # making the directory
            os.makedirs(self.path_save_pics_thumb)
            os.makedirs(self.path_save_pics_eq_hist)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.plotFiltersCheckBox.setChecked(False)
        self.plotObjectsCheckBox.setChecked(False)
        self.plotFiltersCheckBox_3.setChecked(False)
        self.plotObjectsCheckBox_3.setChecked(False)


        if self.useOriginalCheckBox.isChecked():
            self.objectsFeaturesThread.start()
            self.objectsFeaturesThread.set_args(
                                self.path_read_thumb, multiprocess, self.path_save_pics_thumb, self.cpuCoresSpinBox.value(),
                                {'features': self.featuresToRun, 'plot_feature': False},
                                channel=self.channelComboBox.currentText().lower(),
                                noise_filter=self.noiseComboBox.currentText().lower(),
                                edge_detection=self.edgeComboBox.currentText().lower(),
                                noise_filter2=self.noise2ComboBox.currentText().lower(),
                                thresholding=self.thresholdingComboBox.currentText().lower(),
                                closing=self.binClosingComboBox.currentText().lower(),
                                fill_holes=self.fillingHolesComboBox.currentText().lower(),
                                filter_params=[None, (self.noiseSpinbox.value()),
                                      (self.edgeSpinbox.value()),
                                      (self.noise2Spinbox.value()),
                                      (self.bin_closingSpinbox.value()),
                                      (self.thresholding_minSpinbox.value()), self.background_img, 0],
                               plot_filters=self.plotFiltersCheckBox.isChecked(),
                               plot_object=self.plotObjectsCheckBox.isChecked(),
                               min_region_size=(self.minRegionSpinbox.value()), cropped_image_save=True)

            warnings.filterwarnings("ignore")
            self.objectsFeaturesThread.started.connect(self.first_threadStarted)

            if self.useContrastCheckBox.isChecked() == False:
                warnings.filterwarnings("ignore")
                self.objectsFeaturesThread.finished.connect(self.threadFinished)



        if self.useContrastCheckBox.isChecked():

            if self.useOriginalCheckBox.isChecked():
                warnings.filterwarnings("ignore")
                self.objectsFeaturesThread.finished.connect(self.objectsFeaturesThread2.start)

            else:
                self.objectsFeaturesThread2.start()

            self.objectsFeaturesThread2.set_args(
                self.path_read_eq_hist, multiprocess, self.path_save_pics_eq_hist, self.cpuCoresSpinBox.value(),
                {'features': self.featuresToRun, 'plot_feature': False},
                channel=self.channelComboBox_2.currentText().lower(),
                noise_filter=self.noiseComboBox_2.currentText().lower(),
                edge_detection=self.edgeComboBox_2.currentText().lower(),
                noise_filter2=self.noise2ComboBox_2.currentText().lower(),
                thresholding=self.thresholdingComboBox_2.currentText().lower(),
                closing=self.binClosingComboBox_2.currentText().lower(),
                fill_holes=self.fillingHolesComboBox_2.currentText().lower(),
                filter_params=[None, (self.noiseSpinbox_2.value()),
                               (self.edgeSpinbox_2.value()),
                               (self.noise2Spinbox_2.value()),
                               (self.bin_closingSpinbox_2.value()),
                               (self.thresholding_minSpinbox_2.value()), self.background_img, 1],
                plot_filters=self.plotFiltersCheckBox_3.isChecked(),
                plot_object=self.plotObjectsCheckBox_3.isChecked(),
                min_region_size=(self.minRegionSpinbox_2.value()), cropped_image_save=True)

            warnings.filterwarnings("ignore")
            self.objectsFeaturesThread2.finished.connect(self.threadFinished)
            self.objectsFeaturesThread2.started.connect(self.second_threadStarted)

    def threadFinished(self):
        self.statusbar.clearMessage()
        self.plotObjectsCheckBox_3.setChecked(True)
        self.plotObjectsCheckBox.setChecked(True)
        QMessageBox.information(self, 'Processing', 'Processing finished.')

    def first_threadStarted(self):
        self.statusbar.showMessage('Processing thumb photos...')

    def second_threadStarted(self):
        self.statusbar.showMessage('Processing eq_hist photos...')

    def show_image(self):

        # check if directory has been stated by the user
        try:
            self.sites_thumb
            self.sites_eq_hist
        except:
            return
        else:
            #If directory exists -> directory picture
            if int(self.imageNumSpinbox.value()) in self.sites_thumb:
                file_read = os.path.join(self.path_read_thumb, "site{}.jpg".format(str(self.imageNumSpinbox.value())))
                self.item.setPixmap(QPixmap(file_read))
            else:  #else  -> no image
                self.item.setPixmap(QPixmap())
            self.scene.setSceneRect(self.item.boundingRect())
            self.image.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

            #If directory exists -> directory picture
            if int(self.imageNumSpinbox.value()) in self.sites_eq_hist:
                file_read = os.path.join(self.path_read_eq_hist, "site{}.jpg".format(str(self.imageNumSpinbox.value())))
                self.item2.setPixmap(QPixmap(file_read))
            else:  #else  -> no image
                self.item2.setPixmap(QPixmap())
            self.scene2.setSceneRect(self.item2.boundingRect())
            self.image2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)

    def openFileNamesDialog(self):

        sender = self.sender()
        if sender.text() == 'Pictures folder':
            path_read = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if len(path_read) == 0:
                if ('image_paths' in dir(self)) and ('sites_thumb' in dir(self)):
                    self.afterDetectionStatAct.setChecked(True)
                else:
                    self.afterDetectionStatAct.setChecked(False)
                return

            # Do not overwrite the existing folder
            # Add +1 to it's name
            all_files_in_directory = os.listdir(path_read)
            wanted_names = [i for i in all_files_in_directory if i[:7] == 'Objects']

            if len(wanted_names) == 0:
                path_save_pics = os.path.join(path_read, 'Objects')

            else:
                wanted_numbers = [int((i.split('s')[1])) if i != 'Objects' else 1 for i in wanted_names]
                max_number = max(wanted_numbers)
                path_save_pics = os.path.join(path_read, 'Objects{}'.format(max_number + 1))

            self.path_read_thumb = os.path.join(path_read, 'thumb')
            self.path_save_pics_thumb = os.path.join(path_save_pics,'thumb')
            self.path_read_eq_hist = os.path.join(path_read, 'eq_hist')
            self.path_save_pics_eq_hist = os.path.join(path_save_pics,'eq_hist')

            try:
                #Check if selected file does have 'site...'
                self.image_paths = [path for path in os.listdir(self.path_read_thumb) if path.startswith('site')]
                self.sites_thumb = [int(re.findall(r'\d+', path.split(os.sep)[-1])[0]) for path in self.image_paths]
                self.image_paths = [path for path in os.listdir(self.path_read_eq_hist) if path.startswith('site')]
                self.sites_eq_hist = [int(re.findall(r'\d+', path.split(os.sep)[-1])[0]) for path in self.image_paths]

            except FileNotFoundError:
                self.afterDetectionStatAct.setChecked(False)
                QMessageBox.information(self, 'Incorrect Image folder', "Incorrect image folder. Please re-select image folder.")
                return

            self.afterDetectionStatAct.setChecked(True)
            # Set min max
            self.imageNumSpinbox.setMaximum(max(self.sites_thumb))
            self.imageNumHorizontalSlider.setMaximum(max(self.sites_thumb))
            self.imageNumSpinbox.setMinimum(min(self.sites_thumb))
            self.imageNumHorizontalSlider.setMinimum(min(self.sites_thumb))
            self.show_image()

        if sender.text() == 'MD file':

            path_read = (QFileDialog.getOpenFileName(self, "Select MD file"))[0]
            # turn off checkBoxes if user presses 'cancel' and mark if user selects the file format
            if len(path_read) == 0:
                if 'df_original' in dir(self):
                    self.MDStatAct.setChecked(True)
                else:
                    self.MDStatAct.setChecked(False)
                return
            else:
                MD_address = path_read
                if 'MD' in MD_address:
                    try:
                        df = pd.read_csv(MD_address, decimal=',', sep='\t')
                    except FileNotFoundError:
                        QMessageBox.information(self, 'Incorrect MD file selected',
                                                "Incorrect MD file selected. Please re-select MD file - the filename must have MD string")
                        self.MDStatAct.setChecked(False)
                        return

                    self.MDStatAct.setChecked(True)
                    #Get the address of the site with smallest fluence
                    site_nr = df.loc[df['Fluency Mean, J/cm2'].idxmin(), 'Site Nr.']
                    min_fluence = df['Fluency Mean, J/cm2'].idxmin()
                    #self.background_img = skimage.io.imread(os.path.join(self.path_read_thumb, 'site{}.jpg'.format(site_nr)))[::15, ::15, :]
                    #self.background_img = skimage.io.imread(os.path.join(self.path_read_thumb, 'site{}.jpg'.format(13)))[::12, ::12, :]
                else:
                    QMessageBox.information(self, 'Incorrect MD file selected',
                                            "Incorrect MD file selected. Please re-select MD file - the filename must have 'MD' letters")
                    self.MDStatAct.setChecked(False)
                    return

        if sender.text() == 'Objects folder':

            path_read = QFileDialog.getExistingDirectory(self, "Select pictures directory")

            if len(path_read) == 0:
                if 'all_objects_thumb' in dir(self):
                    self.objectsStatAct.setChecked(True)
                else:
                    self.objectsStatAct.setChecked(False)
                return
            else:
                self.afterDetection_address = path_read
                try:
                    self.all_objects_thumb = pd.read_csv(
                        os.path.join(os.path.join(self.afterDetection_address, 'thumb'), 'directory pictures data.csv'))
                except FileNotFoundError:
                    QMessageBox.information(self, 'Incorrect Objects folder',
                                            "Incorrect Objects folder. Please re-select Objects folder - the folder which was made after Image Processing")
                    self.objectsStatAct.setChecked(False)
                    return
                self.objectsStatAct.setChecked(True)

        if sender.text() == 'Pictures folder 2':

            path_read = (QFileDialog.getExistingDirectory(self, "Select Directory"))

            if len(path_read) == 0:
                if ('image_paths' in dir(self)) and ('sites_thumb' in dir(self)):
                    self.afterDetectionStatAct2.setChecked(True)
                else:
                    self.afterDetectionStatAct2.setChecked(False)

            self.path_read_thumb2 = os.path.join(path_read, 'thumb')
            self.path_read_eq_hist2 = os.path.join(path_read, 'eq_hist')

            try:
                # Check if selected file does have 'site...'
                self.image_paths2 = [path for path in os.listdir(self.path_read_thumb2) if path.startswith('site')]
                self.sites_thumb2 = [int(re.findall(r'\d+', path.split(os.sep)[-1])[0]) for path in self.image_paths2]
                self.image_paths2 = [path for path in os.listdir(self.path_read_eq_hist2) if path.startswith('site')]
                self.sites_eq_hist2 = [int(re.findall(r'\d+', path.split(os.sep)[-1])[0]) for path in self.image_paths2]

            except FileNotFoundError:
                self.afterDetectionStatAct2.setChecked(False)
                QMessageBox.information(self, 'Incorrect Image folder',
                                        "Incorrect image folder. Please re-select image folder.")
                return
            self.afterDetectionStatAct2.setChecked(True)


class GenericThread(QThread):
   finished = pyqtSignal()

   def __init__(self, *args, **kwargs):
       super(GenericThread, self).__init__()
       self.args = args
       self.kwargs = kwargs

   def set_args(self, *args, **kwargs):
       self.args = args
       self.kwargs = kwargs

class ObjectsFeaturesThread(GenericThread):
   # def __init__(self):
   #     self.all_objects_thumb = None

    def run(self):
       self.all_objects_thumb = identification.get_regions_from_images_mp(*self.args, **self.kwargs)
       self.finished.emit()

    #def get_dataframe(self):
    #    return self.all_objects_thumb

class ObjectsFeaturesThread2(GenericThread):
    #def __init__(self):
    #    self.all_objects_eq_hist = None

    def run(self):
       self.all_objects_eq_hist = identification.get_regions_from_images_mp(*self.args, **self.kwargs)
       self.finished.emit()

   # def get_dataframe(self):
    #    return self.all_objects_eq_hist

class Canvas(FingureCanvas):
    def __init__(self, MainWindow, img, patches, width =5, height = 5):

        fig = Figure(figsize=(width, height))
        self.ax = fig.add_subplot(111)

        FingureCanvas.__init__(self, fig)
        #self.setParent(MainWindow)
        self.plot(img, patches)
        fig.tight_layout()

    def plot(self, img, patches):

        ax = self.figure.add_subplot(111)
        ax.imshow(img.astype(np.uint8))

        for p in patches:
            new_p = copy(p)
            ax.add_patch(new_p)

        ax.set_axis_off()



