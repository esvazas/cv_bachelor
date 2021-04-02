from PyQt5.QtWidgets import *
import numpy as np
import sklearn.metrics
from gui.info_clustering_dialog_ui import Ui_Dialog

class clustering_Dialog(QDialog, Ui_Dialog):
    
    def __init__(self, df, all_objects_thumb):
        super().__init__()
        self.setupUi()

        self.df = df
        self.all_objects_thumb = all_objects_thumb

        self.thumbAccuracy_lineEdit.setReadOnly(True)
        self.thumbDD_lineEdit_percents.setReadOnly(True)
        self.thumbDS_lineEdit_percents.setReadOnly(True)
        self.thumbSS_lineEdit_percents.setReadOnly(True)
        self.thumbSD_lineEdit_percents.setReadOnly(True)

        self.eq_histAccuracy_lineEdit.setReadOnly(True)
        self.eq_histDD_lineEdit_percents.setReadOnly(True)
        self.eq_histDS_lineEdit_percents.setReadOnly(True)
        self.eq_histSS_lineEdit_percents.setReadOnly(True)
        self.eq_histSD_lineEdit_percents.setReadOnly(True)

        self.setupSignals()

    def setupSignals(self):
        self.calculatePushButton.clicked.connect(self.calculate_thumb_accuracy)
        self.damagedOutliersLineEdit.editingFinished.connect(self.text_changed)

    def setClusteringInfo_originalPics(self, points, clusters, outliers):
        self.points_lineEdit.setText(str(points))
        self.clusters_lineEdit.setText(str(clusters))
        self.outliers_lineEdit.setText(str(outliers))

    def setClusteringInfo_highContrastPics(self, points, clusters, outliers):
        self.points_lineEdit2.setText(str(points))
        self.clusters_lineEdit2.setText(str(clusters))
        self.outliers_lineEdit2.setText(str(outliers))

    def setupUi(self):
        super().setupUi(self)

    def text_changed(self):

        try:
            self.thumb_damage_label = self.damagedOutliersLineEdit.text().split(",")
            self.thumb_damage_label = list(map(int, self.thumb_damage_label))
        except ValueError:
            pass



    def setThumbClusteringDetails (self, acc, dd, ds, ss, sd):
        #dd and etc. are lists


        self.thumbAccuracy_lineEdit.setText(str(acc))

        self.thumbDD_lineEdit_percents

        self.thumbDD_lineEdit_percents.setText(str(dd[0]))
        self.thumbDS_lineEdit_percents.setText(str(ds[0]))
        self.thumbSS_lineEdit_percents.setText(str(ss[0]))
        self.thumbSD_lineEdit_percents.setText(str(sd[0]))

        self.thumbDD_lineEdit_number.setText("{}/{}".format(str(dd[1]),str(dd[2])))
        self.thumbDS_lineEdit_number.setText("{}/{}".format(str(ds[1]),str(ds[2])))
        self.thumbSS_lineEdit_number.setText("{}/{}".format(str(ss[1]),str(ss[2])))
        self.thumbSD_lineEdit_number.setText("{}/{}".format(str(sd[1]),str(sd[2])))

    def setEq_histClusteringDetails(self, acc, dd, ds, ss, sd):

        self.eq_histAccuracy_lineEdit.setText(str(acc))

        self.eq_histDD_lineEdit_percents.setText(str(dd[0]))
        self.eq_histDS_lineEdit_percents.setText(str(ds[0]))
        self.eq_histSS_lineEdit_percents.setText(str(ss[0]))
        self.eq_histSD_lineEdit_percents.setText(str(sd[0]))

        self.eq_histDD_lineEdit_number.setText("{}/{}".format(str(dd[1]),str(dd[2])))
        self.eq_histDS_lineEdit_number.setText("{}/{}".format(str(ds[1]),str(ds[2])))
        self.eq_histSS_lineEdit_number.setText("{}/{}".format(str(ss[1]),str(ss[2])))
        self.eq_histSD_lineEdit_number.setText("{}/{}".format(str(sd[1]),str(sd[2])))

    def calculate_thumb_accuracy(self):

        self.thumb_damage_label = self.damagedOutliersLineEdit.text().split(",")
        self.thumb_damage_label = list(map(int, self.thumb_damage_label))

        self.damaged_sites_num = []

        self.df_copy = self.df.copy()
        self.df_copy = self.df_copy[['Site Nr.', 'Insp. Status', 'Fluency Mean, J/cm2', 'Insp Pulses passed(F) ']]

        for i in list(np.unique(self.df_copy['Site Nr.'])):
            rows = self.all_objects_thumb.loc[self.all_objects_thumb['site nr'] == (i)]
            site_index = self.df_copy[self.df_copy['Site Nr.'] == i].index[0]  # avoiding problem when the 'Site Nr.' =! indexes
            if len(set(list(rows['PCA labels'])).intersection(set(self.thumb_damage_label))) > 0:
                self.df_copy.loc[int(site_index), 'Fit status'] = 1
                self.damaged_sites_num.append(i)
            else:
                self.df_copy.loc[int(site_index), 'Fit status'] = 0

        pr_score = sklearn.metrics.accuracy_score(self.df_copy['Insp. Status'], self.df_copy['Fit status'])

        acc = (np.round(pr_score, 2))
        dfn = self.df_copy[self.df_copy['Insp. Status'] == 1]
        dd = [np.round(dfn[dfn['Fit status'] == 1].shape[0] / dfn.shape[0], 2), dfn[dfn['Fit status'] == 1].shape[0], dfn.shape[0]]

        dfn = self.df_copy[self.df_copy['Insp. Status'] == 1]
        ds = [(np.round(dfn[dfn['Fit status'] == 0].shape[0] / dfn.shape[0],2)), dfn[dfn['Fit status'] == 0].shape[0], dfn.shape[0]]

        dfn = self.df_copy[self.df_copy['Insp. Status'] == 0]
        ss = [np.round(dfn[dfn['Fit status'] == 0].shape[0] / dfn.shape[0], 2), dfn[dfn['Fit status'] == 0].shape[0], dfn.shape[0]]

        dfn = self.df_copy[self.df_copy['Insp. Status'] == 0]
        sd = [np.round(dfn[dfn['Fit status'] == 1].shape[0] / dfn.shape[0], 2), dfn[dfn['Fit status'] == 1].shape[0], dfn.shape[0]]

        idx = (self.df_copy.loc[:, 'Insp. Status'].astype(int) == self.df_copy.loc[:, 'Fit status']).astype(int)
        self.df_copy['Accuracy'] = idx.astype(int)

        self.setThumbClusteringDetails(acc, dd, ds, ss, sd)

    def get_damagedLabels(self):
        return self.thumb_damage_label




