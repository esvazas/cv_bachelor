import os, re, sys
import skimage.io
import matplotlib.pyplot as plt
import errno

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from functools import partial
import pyqtgraph as pg

from gui.features_dialog2 import FeaturesDialog2
from gui.info_clustering_dialog import clustering_Dialog
from gui.radius_dialog import ShowGraph
from gui.cluster_groupBox import ClusterGroupBox
from gui.outliers_dialog import Outliers
from gui.object_recognition_mainwindow_ui import Ui_MainWindow

import pandas as pd
import numpy as np
import sklearn.cluster
import sklearn.metrics
import sklearn.preprocessing
import logic.recognition as recognition
import skimage.io
import warnings
import matplotlib
import corner




class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setupSignals()
        self.pca_checkButton = True
        self.pca_checkButton2 = True

        self.default_features = ['Area/Circle diff', 'Area/Square diff', 'Blue part',
         'Blue part eq_hist', 'Centroids column', 'Centroids row',
         'Color inertia', 'Color inertia eq_hist', 'Eccentricity', 'Entropy max',
         'Entropy max eq_hist', 'Entropy mean', 'Entropy mean eq_hist',
         'Entropy min', 'Entropy min eq_hist', 'Entropy std',
         'Entropy std eq_hist', 'Fifth moment', 'Fifth moment eq_hist',
         'First moment', 'First moment eq_hist', 'Fourth moment',
         'Fourth moment eq_hist', 'Gray Max', 'Gray Max eq_hist', 'Gray Mean',
         'Gray Mean eq_hist', 'Gray Min', 'Gray Min eq_hist', 'Gray Std',
         'Gray Std eq_hist', 'Green part', 'Green part eq_hist',
                                 'Object Perimeter/Circle Perimeter',
                                 'Orientation', 'Red part', 'Red part eq_hist',
         'Second moment', 'Second moment eq_hist', 'Seventh moment',
         'Seventh moment eq_hist', 'Shapes Area', 'Shapes Perimeter',
         'Sixth moment', 'Sixth moment eq_hist', 'Sobel max',
         'Sobel max eq_hist', 'Sobel mean', 'Sobel mean eq_hist', 'Sobel std',
         'Sobel std eq_hist', 'Texture ASM', 'Texture ASM eq_hist',
         'Texture contrast', 'Texture contrast eq_hist', 'Texture correlation',
         'Texture correlation eq_hist', 'Texture dissimilarity',
         'Texture dissimilarity eq_hist', 'Texture energy',
         'Texture energy eq_hist', 'Texture homogeneity',
         'Texture homogeneity eq_hist', 'Third moment', 'Third moment eq_hist']

        self.default_features2 = ['Area/Circle diff', 'Area/Square diff', 'Blue part',
         'Blue part thumb', 'Centroids column', 'Centroids row',
         'Color inertia', 'Color inertia thumb', 'Eccentricity', 'Entropy max',
         'Entropy max thumb', 'Entropy mean', 'Entropy mean thumb',
         'Entropy min', 'Entropy min thumb', 'Entropy std',
         'Entropy std thumb', 'Fifth moment', 'Fifth moment thumb',
         'First moment', 'First moment thumb', 'Fourth moment',
         'Fourth moment thumb', 'Gray Max', 'Gray Max thumb', 'Gray Mean',
         'Gray Mean thumb', 'Gray Min', 'Gray Min thumb', 'Gray Std',
         'Gray Std thumb', 'Green part', 'Green part thumb', 'Object Perimeter/Circle Perimeter',
                                  'Orientation',  'Red part', 'Red part thumb',
         'Second moment', 'Second moment thumb', 'Seventh moment',
         'Seventh moment thumb', 'Shapes Area', 'Shapes Perimeter',
         'Sixth moment', 'Sixth moment thumb', 'Sobel max',
         'Sobel max thumb', 'Sobel mean', 'Sobel mean thumb', 'Sobel std',
         'Sobel std thumb', 'Texture ASM', 'Texture ASM thumb',
         'Texture contrast', 'Texture contrast thumb', 'Texture correlation',
         'Texture correlation thumb', 'Texture dissimilarity',
         'Texture dissimilarity thumb', 'Texture energy',
         'Texture energy thumb', 'Texture homogeneity',
         'Texture homogeneity thumb', 'Third moment', 'Third moment thumb']

        self.additional_features = ['site nr', 'Object in image nr', 'Min row', 'Max row', 'Max column', 'Min column']
        self.delete_centroids_column = 0
        self.delete_centroids_row = 0
        self.delete_shapes_area = 0

        self.features_was_opened = 0 #If User opens Features section, he sets the features, otherwise we need to set the default features to open
        self.image_to_show = 0 #contains the number of image to show if the User presses >>> or <<<

        self.directory_cropped_image = None
        self.directory_images = None

        self.minimalSizeObjects_thumb = 0
        self.minimalSizeObjects_eqhist = 0
        self.maximalSizeObjects_thumb = 0
        self.maximalSizeObjects_eqhist = 0
        self.wanted_sizeObjects_thumb = 0

        #Default values of comboBoxes, checkBoxes and spinBoxes
        self.radius_doubleSpinBox.setValue(150)

        # Slider Max range(0,10000) ir tai atitinka spinbox max range (0,10)
        self.eps_doubleSpinBox.setValue(1.0)
        self.eps_horizontalSlider.setValue(100)

        self.min_pts_doubleSpinBox.setValue(3)

        self.eps_doubleSpinBox2.setValue(1.0)
        self.min_pts_doubleSpinBox2.setValue(3)

        self.clustersNumberToShowSpinBox.setValue(5)
        self.clustersNumberToShowSpinBox.setToolTip("Pictures number /n Number of central images in each cluster which will be visualised.")
        self.clustersNumberToShowSpinBox.setMinimum(1)

        self.radiusAllPictureCheckBox.setChecked(False)
        self.radiusAllPictureCheckBox.setToolTip("All objects in the image are taken. /n No limitations on objects coordinates are added.")

        #self.preProcess_pushButton.setEnabled(False)

        self.bar = self.menuBar()
        self.Files = self.bar.addMenu('Files')

        self.afterDetectionStatAct = QAction('Objects folder', self, checkable=True)
        self.afterDetectionStatAct.setChecked(False)
        self.afterDetectionStatAct.triggered.connect(self.openDirectoryDialog)
        self.Files.addAction(self.afterDetectionStatAct)

        self.MDStatAct = QAction('MD file', self, checkable=True)
        self.MDStatAct.setChecked(False)
        self.MDStatAct.triggered.connect(self.openFileNamesDialog)
        self.Files.addAction(self.MDStatAct)


        self.featuresMenuBar = self.bar.addMenu('Features')
        self.featuresRunStatAct = QAction("Select features", self)
        self.featuresRunStatAct.triggered.connect(self.openFeaturesDialog)
        self.featuresMenuBar.addAction(self.featuresRunStatAct)

        self.analysisPlotMenuBar = self.bar.addMenu('Analysis')
        self.scatterPlotStatAct = QAction("Scatter plot", self)
        self.scatterPlotStatAct.triggered.connect(self.make_scatterPlot)
        self.analysisPlotMenuBar.addAction(self.scatterPlotStatAct)
        self.cornerPlotStatAct = QAction("Corner plot", self)
        self.cornerPlotStatAct.triggered.connect(self.make_cornerPlot)
        self.analysisPlotMenuBar.addAction(self.cornerPlotStatAct)
        self.accuracyPlotStatAct = QAction("Classification accuracy", self)
        self.accuracyPlotStatAct.triggered.connect(self.calculate_accuracy)
        self.analysisPlotMenuBar.addAction(self.accuracyPlotStatAct)

        self.formFileMenuBar = self.bar.addMenu('Form Files')
        self.formMD_ClusterLabelsRunStatAct = QAction("MD file: Cluster labels -> Insp. Status", self)
        self.formMD_ClusterLabelsRunStatAct.triggered.connect(partial(self.form_MDfile, 'C'))
        self.formFileMenuBar.addAction(self.formMD_ClusterLabelsRunStatAct)

        self.formMD_DamageLabelsRunStatAct = QAction("MD file: Damaged labels -> Insp. Status", self)
        self.formMD_DamageLabelsRunStatAct.triggered.connect(partial(self.form_MDfile, 'D'))
        self.formFileMenuBar.addAction(self.formMD_DamageLabelsRunStatAct)

        self.formClusterFoldersRunStatAct = QAction("Cluster folders", self)
        self.formClusterFoldersRunStatAct.triggered.connect(self.form_clusterFolder)
        self.formFileMenuBar.addAction(self.formClusterFoldersRunStatAct)

    def setupSignals(self):

        self.preProcessPushButton.clicked.connect(self.preprocessing)
        self.showClustersPushButton.clicked.connect(self.clustering)
        #self.radiusPushButton.clicked.connect(self.show_radius)
        self.mergeClustersPushButton.clicked.connect(self.merge_clusters)
        self.labelOutliersPushButton.clicked.connect(self.label_outliers)
        self.eps_horizontalSlider.valueChanged.connect(self.slider_value_changed)
        self.eps_doubleSpinBox.valueChanged.connect(self.spinbox_value_changed)
        self.radiusAllPictureCheckBox.toggled.connect(self.radius_changed)
        self.clustersNumberToShowSpinBox.valueChanged.connect(self.show_clusters)

    def slider_value_changed(self):

        self.eps_doubleSpinBox.setValue(self.eps_horizontalSlider.value()/100)

    def spinbox_value_changed(self):

        self.eps_horizontalSlider.setValue(self.eps_doubleSpinBox.value()*100)

        if (len(self.afterDetection_address) != 0) and ('X_train_pca' in dir(self)):
            # User specified the specific information for files_location + pre-processing has been done
            self.clustering()

    def radius_changed(self, state):

        if state and (self.directory_images != None):
            img = os.path.join(self.directory_images, (os.listdir(self.directory_images))[0])
            pic = skimage.io.imread(img)
            image_radius = min(pic.shape[0], pic.shape[1])

            self.radius_doubleSpinBox.setValue(image_radius)
            self.radius_doubleSpinBox.setEnabled(False)

        else:
            self.radius_doubleSpinBox.setEnabled(True)


    def form_MDfile(self, case):

        # To get the image size (and it's middle point coordinates):
        # -> Reading the first element in the directory

        delete1 = 0 # Delete the centroids rows/columns columns if they did not exist
        delete2 = 0

        img = os.path.join(self.directory_images, (os.listdir(self.directory_images))[0])
        pic = skimage.io.imread(img)
        image_width, image_height = pic.shape[0], pic.shape[1]

        df_withLabels = self.df_original.copy()
        image_center = np.array([image_width / 2, image_height / 2])

        #Add centroids rows and columns to dataframe if it do not exist
        if 'Centroids row' not in list(self.all_objects_thumb.columns):
            self.all_objects_thumb['Centroids row'] = self.centroids_coords['Centroids row']
            delete1 = 1

        if 'Centroids column' not in list(self.all_objects_thumb.columns):
            self.all_objects_thumb['Centroids column'] = self.centroids_coords['Centroids column']
            delete2 = 1

        if 'thumb_damage_label' not in dir(self):
            # If variable does not exist, rows with -1 will be filled
            self.thumb_damage_label = []

        for i in df_withLabels.index:

            site_nr = df_withLabels.loc[i, 'Site Nr.']
            rows = self.all_objects_thumb[self.all_objects_thumb['site nr'] == site_nr]
            rows_coordinates = rows[['Centroids column', 'Centroids row']]
            label = list(rows['PCA labels'])
            any_damage = set(label).intersection(set(self.thumb_damage_label))
            damaged = 1 if len(any_damage) > 0 else 0

            if len(rows) != 0: # site does have object, now get the number of objects

                if len(rows) >= 2: # -> Site has got multiple objects
                    closest_idx = np.array(
                        np.sqrt(np.sum((np.absolute(np.subtract(image_center, rows_coordinates)) ** 2), axis=1))).argmin()
                    label_to_set = label[closest_idx]

                else: # -> Site has got one object
                    label_to_set = label[0]

                if case == 'C':
                    df_withLabels.loc[i, 'Insp. Status'] = label_to_set

                elif case == 'D':
                    df_withLabels.loc[i, 'Insp. Status'] = damaged

            else:  # -> Site does not have any objects

                if case == 'C':
                    df_withLabels.loc[i, 'Insp. Status'] = -1
                elif case == 'D':
                    df_withLabels.loc[i, 'Insp. Status'] = 0

        name = self.MD_address.split(os.sep)[-1]
        name = "{}_v2.xls".format(name.split('.')[0])
        df_withLabels.to_csv(os.path.join(self.directory_cropped_image, name), index=False, decimal=',', sep='\t')
        print("File is saved in: ", os.path.join(self.directory_cropped_image, name))

        if delete1:
            del self.all_objects_thumb['Centroids row']

        if delete2:
            del self.all_objects_thumb['Centroids column']

    def form_clusterFolder(self):

        print("Cluster")

        names = ['thumb']
        method_number = 0

        for name in names:
            warnings.filterwarnings("ignore")

            directory_save_cluster = os.path.join(self.directory_cropped_image, 'Cluster')

            if name == 'thumb':
                all_objects2 = self.all_objects_thumb.copy()
                directory_img = os.path.join(self.directory_cropped_image,'')

            # elif name == 'eq_hist':
            #    all_objects2 = clustering_eqhist.copy()
            #    directory_img = directory_cropped_image_eq_hist

            print("Directory to save: %s" % directory_img)

            # Check if damaged labels are already created
            # If variable does not exist, rows with -1 will be filled
            if 'thumb_damage_label' not in dir(self):
                self.thumb_damage_label = []

            for i in list(np.unique(all_objects2['PCA labels'])):
                print("Label being saved now: %s" % (i))
                if i in self.thumb_damage_label:
                    directory_save_cluster_cluster = "{}{} damaged".format(directory_save_cluster,i)
                else:
                    directory_save_cluster_cluster = "{}{} survived".format(directory_save_cluster,i)

                try: # Make the directory to save images
                    os.makedirs(directory_save_cluster_cluster)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise

                rows = all_objects2.loc[all_objects2['PCA labels'] == (i)]
                site_nr = list(rows['site nr'])
                img_nr = list(rows['Object in image nr'])

                for img in range(len(site_nr)):
                    site = str(site_nr[img]) + '_' + str(int(img_nr[img]))
                    img = skimage.io.imread(directory_img + "cropped_site{}.png".format(site))
                    skimage.io.imsave(os.path.join(directory_save_cluster_cluster, 'cropped_site{}.png'.format(site)), img)

    def calculate_accuracy(self):

        clusteringAccuracy = clustering_Dialog(self.df, self.all_objects_thumb)

        if clusteringAccuracy.exec():
            self.thumb_damage_label = clusteringAccuracy.get_damagedLabels()

        else:
            # Cancel
            clusteringAccuracy.close()

    def make_cornerPlot(self):

        if self.pca_checkButton:
            figure = corner.corner(self.X_train_pca)
            plt.show(figure)

        else:
            figure = corner.corner(self.X_train_pca, labels=list(self.X_train_labels), quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
            plt.show(figure)


    def make_scatterPlot(self):

        labels = self.all_objects_thumb[['PCA labels']].astype(int).iloc[:, 0].tolist()
        self.corner_scatter(self.X_train_pca, scatter_kwargs={'c': labels})

    def corner_scatter(self, data_array, scatter_kwargs=None, hist_kwargs=None):
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
        hist_kwargs = {} if hist_kwargs is None else hist_kwargs

        n_samples, n_dim = data_array.shape

        fig = plt.figure(figsize=(12, 8))
        fig, axes = plt.subplots(nrows=n_dim, ncols=n_dim)
        # Plot scatters
        for col in range(n_dim - 1):
            x = col
            y = x + 1
            for row in list(range(col + 1, n_dim)):
                axes[row, col].scatter(data_array[..., x], data_array[..., y], **scatter_kwargs)
                y += 1

        # Plot histograms
        for i in range(n_dim):
            axes[i, i].hist(data_array[..., i], **hist_kwargs)

        fig.show()

    def label_outliers(self):


        if len(self.sites) != 0:

            rows = self.all_objects_thumb[self.all_objects_thumb['PCA labels'] == -1]
            coordinates = np.array(self.centroids_coords[self.centroids_coords.index.isin(rows.index)])
            rectangles = self.objects_rectangles[self.objects_rectangles.index.isin(rows.index)]
            outlierDialog = Outliers(self.directory_cropped_image, self.directory_images, self.sites, self.site_address, coordinates, rectangles)

            if outlierDialog.exec():
                new_outliers_labels = outlierDialog.get_newOutliersLabels()

                for counter,idx in enumerate(rows.index):
                    self.all_objects_thumb.loc[idx, 'PCA labels'] = new_outliers_labels[counter]

                if -1 in new_outliers_labels:
                    #User did not specified all outliers
                    idx = [index for index, value in enumerate(new_outliers_labels) if value == -1]

                    values = np.array(self.outliers_sites)
                    self.outliers_sites = list(values[idx])

                    values = np.array(self.sites)
                    self.sites = list(values[idx])

                else:
                    #User divided all outliers to the classes
                    self.outliers_sites = []

                self.show_clusters()

            else:
                # Cancel
                outlierDialog.close()
        else:
            # Condition -> No outliers
            return

    def show_radius(self):

        try:
            dirs = os.listdir(self.directory_images)
            first_element = dirs[0].split('.')[0]
            img_shape = skimage.io.imread("{}{}.jpg".format(dir, first_element)).shape

        except AttributeError:
            img_shape = (1000,1000,3)

        radius = ShowGraph(img_shape)


        if radius.exec():
            print("super")
        else:
            #Cancel
            radius.close()

    def merge_clusters(self):

        self.spinboxes_values = [i.getSpinBox() for i in self.cluster_images]
        clusters = list(np.unique(self.all_objects_thumb['PCA labels']))

        # If outliers exist:
        # -> clusters = [0,1,2,....,-1]
        # Else:
        # -> clusters = [0,1,2,3...]

        if clusters[0] == -1:
            clusters.insert(len(clusters), clusters.pop(0))

        for i in range(len(self.spinboxes_values)):
            if clusters[i] != self.spinboxes_values[i]:
                rows = self.all_objects_thumb[self.all_objects_thumb['PCA labels'] == clusters[i]]
                for obj in list(rows.index):
                    self.all_objects_thumb.loc[obj, 'PCA labels'] = self.spinboxes_values[i]

        clusters = list(np.unique(self.all_objects_thumb['PCA labels']))
        clusters_range = list(range(max(clusters)+1))
        missing_element = list(set(clusters_range).difference(set(clusters)))

        for miss_element in missing_element[::-1]:

            rows = self.all_objects_thumb[self.all_objects_thumb['PCA labels'] > miss_element]
            for obj in list(rows.index):
                self.all_objects_thumb.loc[obj, 'PCA labels'] = self.all_objects_thumb.loc[obj, 'PCA labels'] - 1

        # Show new clusters
        self.show_clusters()

    def clustering(self):

        if self.X_train_pca.shape[0] != 0 and float(self.eps_doubleSpinBox.value()) != 0:

            self.outliers_sites, self.all_objects_thumb, self.sites = recognition.DBSCAN_get_outliers(self.X_train_pca,
                                                                                                  self.all_objects_thumb,
                                                                                                  eps=self.eps_doubleSpinBox.value(),
                                                                                                  min_samples=self.min_pts_doubleSpinBox.value())

            # Opens information dialog window of clusters, points and outliers number
            self.openClusteringInfoDialog()

            # Show new clusters
            self.show_clusters()

    def clearWidget(self, layout):

         while layout.count():
             child = layout.takeAt(0)
             if child.widget():
                 child.widget().deleteLater()

    def show_clusters(self):

        self.cluster_images = [] #contains all ClusterGroupBox objets

        # Calculate Euclidean distance and return the main points of each cluster
        clustering, similar_pics, max_number_clusters = recognition.updated_DBSCAN_clustering(self.X_train_pca,
                                                                         self.all_objects_thumb,
                                                                         k_number=self.clustersNumberToShowSpinBox.value())

        cluster_number, clusters_to_visualise = len(similar_pics) - 1, len(similar_pics[0])  # kiek klasteriu ir kiek jame paveikslu

        self.clustersNumberToShowSpinBox.setMaximum(max_number_clusters)
        self.clustersNumberToShowSpinBox.setValue(clusters_to_visualise)

        #if clusters_to_visualise > self.clustersNumberToShowSpinBox.value():
        #    similar_pics = [y[:self.clustersNumberToShowSpinBox.value()] for y in similar_pics]

        clustering['Object in image nr'] = clustering['Object in image nr'].astype(int)
        clustering['site nr'] = clustering['site nr'].astype(int)

        self.site_address = [list(clustering.iloc[i, :]['site nr'].map(str) + '_'
                                  + clustering.iloc[i, :]['Object in image nr'].map(str)) for i in similar_pics]

        # Make sure that Layout is empty
        self.clearWidget(self.horizontalLayout_4)

        if len(self.outliers_sites) != 0:
            # If outliers exist -> indexes_to_iterate = [0,1,2,3,...,-1]
            indexes_to_iterate = list(range(len(self.site_address)-1))
            indexes_to_iterate.append(-1)
        else:
            indexes_to_iterate = list(range(len(self.site_address)))

        min_element, max_element  = min(indexes_to_iterate), max(indexes_to_iterate)
        label_freq = (self.all_objects_thumb['PCA labels'].value_counts().to_dict())

        for i in indexes_to_iterate:
            groupbox = ClusterGroupBox(self.directory_cropped_image,
                                       self.site_address[indexes_to_iterate.index(i)], i,
                                        min_element, max_element, label_freq[i])

            self.cluster_images.append(groupbox)
            self.horizontalLayout_4.addWidget(groupbox)

    def openClusteringInfoDialog(self):

        self.pointsLineEdit.setText(str(self.X_train_pca.shape[0]))
        self.clustersLineEdit.setText(str(len(np.unique(self.all_objects_thumb['PCA labels']))))
        self.outliersLineEdit.setText(str(len(self.all_objects_thumb[self.all_objects_thumb['PCA labels'] == -1])))

    def openFileNamesDialog(self):

        self.path_read = str(QFileDialog.getOpenFileName(self, "Select the MAIN file"))

        #turn off checkBoxes if user presses 'cancel' and mark if user selects the file format
        if len(self.path_read) == 8:
            if 'df_original' in dir(self):
                self.MDStatAct.setChecked(True)
            else:
                self.MDStatAct.setChecked(False)
            return

        else:
            self.MD_address = self.path_read.split('\'')[1]

            if 'MD' in self.MD_address:
                try:
                    self.df_original = pd.read_csv(self.MD_address, decimal=',', sep='\t')
                except FileNotFoundError:
                    QMessageBox.information(self, 'Incorrect MD file selected',
                                            "Incorrect MD file selected. Please re-select MD file - the filename must have MD string")
                    self.MDStatAct.setChecked(False)
                    return
                self.MDStatAct.setChecked(True)

            else:
                QMessageBox.information(self, 'Incorrect MD file selected',
                                        "Incorrect MD file selected. Please re-select MD file - the filename must have 'MD' letters")
                self.MDStatAct.setChecked(False)
                return

    def openDirectoryDialog(self):

        #turn off checkBoxes if user presses 'cancel' and mark if user selects the directory format
        self.path_read = str(QFileDialog.getExistingDirectory(self, "Select pictures directory"))

        if len(self.path_read) == 0:
            if 'all_objects_thumb' in dir(self):
                self.afterDetectionStatAct.setChecked(True)
            else:
                self.afterDetectionStatAct.setChecked(False)
            return

        else:
            self.afterDetection_address = self.path_read

            try:
                self.all_objects_thumb = pd.read_csv(
                    os.path.join(os.path.join(self.afterDetection_address, 'thumb'), 'directory pictures data.csv'))

                #self.all_objects_eq_hist = pd.read_csv(
                #    os.path.join(os.path.join(self.afterDetection_address, 'eq_hist'), 'directory pictures data.csv'))

            except FileNotFoundError:
                QMessageBox.information(self, 'Incorrect Objects folder',
                                        "Incorrect Objects folder. Please re-select Objects folder - the folder which was made after Image Processing")

                self.afterDetectionStatAct.setChecked(False)
                return

            self.afterDetectionStatAct.setChecked(True)

            long_dir, short_name = os.path.split(self.afterDetection_address)[0], \
                                   os.path.split(self.afterDetection_address)[1]
            self.directory_cropped_image = os.path.join(self.afterDetection_address, 'thumb')
            self.directory_images = os.path.join(long_dir, 'thumb')

            # Preprocessing for Eq_hist pictures
            self.directory_cropped_image_eq_hist = os.path.join(self.afterDetection_address, 'eq_hist')
            self.directory_image_eq_hist = os.path.join(long_dir, 'eq_hist')

            # Save original features for the features dialog pop up
            self.original_features_thumb = self.all_objects_thumb[
                self.all_objects_thumb.columns.intersection(self.default_features)].columns
            #self.original_features_eqhist = self.all_objects_eq_hist[
            #    self.all_objects_eq_hist.columns.intersection(self.default_features)].columns

            self.featuresToRun = list(self.original_features_thumb) + self.additional_features
            #self.featuresToRun2 = list(self.original_features_eqhist) + self.additional_features
            self.featuresToRun2 = list(self.original_features_thumb) + self.additional_features

    def openFeaturesDialog(self):

        self.delete_centroids_row = 0
        self.delete_centroids_column = 0
        self.delete_shapes_area = 0

        #User can choose which features to proceed
        # dialog = FeaturesDialog2(self.original_features_thumb,
        #                          self.original_features_eqhist,
        #                          self.featuresToRun,
        #                          self.featuresToRun2,
        #                          self.additional_features,
        #                          self.pca_checkButton,
        #                          self.pca_checkButton2,
        #                          self.minimalSizeObjects_thumb,
        #                          self.minimalSizeObjects_eqhist,
        #                          self.maximalSizeObjects_thumb,
        #                          self.maximalSizeObjects_eqhist)


        #User can choose which features to proceed
        dialog = FeaturesDialog2(self.original_features_thumb,
                                 self.original_features_thumb,
                                 self.featuresToRun,
                                 self.featuresToRun2,
                                 self.additional_features,
                                 self.pca_checkButton,
                                 self.pca_checkButton2,
                                 self.minimalSizeObjects_thumb,
                                 self.minimalSizeObjects_eqhist,
                                 self.maximalSizeObjects_thumb,
                                 self.maximalSizeObjects_eqhist)
        if dialog.exec():
            # Ok
            self.featuresToRun = dialog.getCheckedFeatures()
            self.pca_checkButton = dialog.getCheckedPCA()
            self.featuresToRun2 = dialog.getCheckedFeatures2()
            self.pca_checkButton2 = dialog.getCheckedPCA2()

            self.wanted_sizeObjects_thumb = dialog.getObjectsSize_thumb()
            self.wanted_sizeObjects_eqhist = dialog.getObjectsSize_eqhist()

            self.features_was_opened += 1 #Shows that User has opened features dialog

            #Check if centroids row/column were selected
            #User did not select this feature, so preprocess it and delete it later
            if 'Centroids row' not in self.featuresToRun:
                self.delete_centroids_row = 1

            if 'Centroids column' not in self.featuresToRun:
                self.delete_centroids_column = 1

            if 'Shapes Area' not in self.featuresToRun:
                self.delete_shapes_area = 1

        else:
            # Cancel
            dialog.close()

    def preprocessing(self):

        # Check if addresses are correct and already given
        if 'MD_address' not in dir(self) or 'afterDetection_address' not in dir(self):
            QMessageBox.information(self, 'No required files',
                                    "Please select Objects folder and MD file beforehand.")

            return

        # Re-read files if user change radius and preprocess files again
        self.all_objects_thumb = pd.read_csv(
            os.path.join(os.path.join(self.afterDetection_address, 'thumb'), 'directory pictures data.csv'))

        # Check if centroids row/column were selected
        # User did not select this feature, so preprocess it and delete it later
        if 'Centroids row' not in self.featuresToRun:
            self.featuresToRun.append('Centroids row')

        if 'Centroids column' not in self.featuresToRun:
            self.featuresToRun.append('Centroids column')

        if 'Shapes Area' not in self.featuresToRun:
            self.featuresToRun.append('Shapes Area')

        self.objects_rectangles = self.all_objects_thumb.loc[:, ['site nr', 'Min row', 'Max row', 'Max column', 'Min column', 'Object in image nr']].copy()
        self.objects_rectangles['Objects'] = self.objects_rectangles.apply(
            lambda row: "{}_{}".format(int(row['site nr']), int(row['Object in image nr'])), axis=1)

        # User selected Centroids row&column
        self.all_objects_thumb = self.all_objects_thumb[self.featuresToRun]

        #Allows to follow which rows were deleted
        self.all_objects_thumb['Objects'] = self.all_objects_thumb.apply(
            lambda row: "{}_{}".format(int(row['site nr']), int(row['Object in image nr'])), axis=1)

        if self.radiusAllPictureCheckBox.isChecked() == False: # Take the objects within the circle
            column = self.all_objects_thumb['Centroids column'] - self.all_objects_thumb['Centroids column'].max() / 2
            row = self.all_objects_thumb['Centroids row'] - self.all_objects_thumb['Centroids row'].max() / 2
            self.all_objects_thumb['dist'] = np.sqrt(column ** 2 + row ** 2)
            self.all_objects_thumb = self.all_objects_thumb.loc[(self.all_objects_thumb['dist'] < self.radius_doubleSpinBox.value())]
            del self.all_objects_thumb['dist']

        # Form a dataframe of Centroids columns and rows.
        self.centroids_coords = None
        self.centroids_coords = pd.DataFrame(self.all_objects_thumb[['Centroids column', 'Centroids row']])
        self.centroids_coords = self.centroids_coords.reset_index(drop= True)

        if self.delete_centroids_column:
            self.featuresToRun.remove("Centroids column")
            del self.all_objects_thumb['Centroids column']

        if self.delete_centroids_row:
            self.featuresToRun.remove("Centroids row")
            del self.all_objects_thumb['Centroids row']

        self.df_original = pd.read_csv(self.MD_address, decimal=',', sep='\t')
        self.df = self.df_original.copy()

        # Delete elements with -2 or -1 'Insp. status' and reindex all_objects after
        # Delete 'Insp. Status' == -2 rows from both (df and all_objects)
        self.df = self.df[self.df['Insp. Status'] != -2]
        site_nr = self.df.loc[self.df['Insp. Status'] != -2]['Site Nr.'].tolist()
        self.all_objects = self.all_objects_thumb[self.all_objects_thumb['site nr'].astype(int).isin(site_nr)]
        self.all_objects = self.all_objects_thumb.reset_index(drop=True)

        self.df = self.df[self.df['Insp. Status'] != -1]
        site_nr = self.df.loc[self.df['Insp. Status'] != -1]['Site Nr.'].tolist()
        self.all_objects_thumb = self.all_objects_thumb[self.all_objects_thumb['site nr'].astype(int).isin(site_nr)]
        self.all_objects_thumb = self.all_objects_thumb.reset_index(drop=True)

        # Set row index starting to - to coincide with 'Site Nr.'
        if self.df.index[0] != min(self.df['Site Nr.'].astype(int)):
            self.df.index = self.df.index + 1

        a1 = self.df.loc[:, ['Site Nr.', 'Insp. Status']]
        a1.columns = ['site nr', 'Insp. Status']
        self.all_objects_thumb['site nr'] = pd.to_numeric(self.all_objects_thumb['site nr'])
        self.all_objects_thumb = pd.merge(a1, self.all_objects_thumb, on='site nr')

        if self.delete_shapes_area:

            if np.round(self.minimalSizeObjects_thumb, 2) != np.round(self.wanted_sizeObjects_thumb, 2):
                # Delete those rows
                self.all_objects_thumb = self.all_objects_thumb[self.all_objects_thumb['Shapes Area'] > self.wanted_sizeObjects_thumb]
                self.all_objects_thumb = self.all_objects_thumb.reset_index(drop=True)

            self.minimalSizeObjects_thumb = min(self.all_objects_thumb['Shapes Area'])
            self.maximalSizeObjects_thumb = max(self.all_objects_thumb['Shapes Area'])
            self.featuresToRun.remove("Shapes Area")
            del self.all_objects_thumb['Shapes Area']

        else: #User specified "Shapes Area"

            if np.round(self.minimalSizeObjects_thumb, 2) != np.round(self.wanted_sizeObjects_thumb, 2):
                # Delete those rows
                self.all_objects_thumb = self.all_objects_thumb[self.all_objects_thumb['Shapes Area'] > self.wanted_sizeObjects_thumb]
                self.all_objects_thumb = self.all_objects_thumb.reset_index(drop=True)

            self.minimalSizeObjects_thumb = min(self.all_objects_thumb['Shapes Area'])
            self.maximalSizeObjects_thumb = max(self.all_objects_thumb['Shapes Area'])

        self.objects_rectangles = self.objects_rectangles.loc[self.objects_rectangles['Objects'].isin(list(self.all_objects_thumb['Objects']))]
        self.objects_rectangles = self.objects_rectangles.sort_values(['site nr','Object in image nr'])
        self.objects_rectangles = self.objects_rectangles.reset_index(drop=True)
        del self.all_objects_thumb['Objects']

        X_train = self.all_objects_thumb[self.all_objects_thumb.columns.difference(['site nr', 'Min row', 'Max row', 'Max column', 'Min column',
                                            'Object in image nr', 'RGB clusters number', 'Insp. Status'])]
        self.X_train_labels = X_train.columns

        if self.pca_checkButton:
            if X_train.shape[0] != 0:
                #Check if radius was too small
                # and there is no objects:
                self.X_train_pca, pca_components, pca_explained_variance = recognition.PCA(X_train, plot=False)
            else:
                # Create empty array for future proceeding
                self.X_train_pca =np.array([])
        else:
            if X_train.shape[0] != 0:
                #Check if radius was too small
                # and there is no objects:
                self.X_train_pca = X_train.values
                sc = sklearn.preprocessing.StandardScaler()
                self.X_train_pca = sc.fit_transform(self.X_train_pca)
            else:
                # Create empty array for future proceeding
                self.X_train_pca = np.array([])

        print("X_train_pca", self.X_train_pca.shape)