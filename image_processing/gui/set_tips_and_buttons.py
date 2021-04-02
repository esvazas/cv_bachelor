def setTips(self):
    # Default values of comboBoxes, checkBoxes and spinBoxes
    self.channelComboBox.setCurrentText('Gray equal')
    self.noiseComboBox.setCurrentText('None')
    self.edgeComboBox.setCurrentText('Entropy')
    self.noise2ComboBox.setCurrentText('None')
    self.thresholdingComboBox.setCurrentText('Yen >')
    self.binClosingComboBox.setCurrentText('Closing')
    self.fillingHolesComboBox.setCurrentText('Fill holes')
    self.plotFiltersCheckBox.setChecked(False)
    self.noiseSpinbox.setValue(0)
    self.edgeSpinbox.setValue(10)
    self.noise2Spinbox.setValue(0)
    self.bin_closingSpinbox.setValue(10)
    self.imageNumSpinbox.setValue(1)
    self.imageNumHorizontalSlider.setValue(1)
    self.minRegionSpinbox.setValue(150)
    self.thresholding_minSpinbox.setValue(0)

    # Default values of comboBoxes, checkBoxes and spinBoxes
    self.channelComboBox_2.setCurrentText('Normalised')
    self.noiseComboBox_2.setCurrentText('None')
    self.edgeComboBox_2.setCurrentText('Entropy')
    self.noise2ComboBox_2.setCurrentText('None')
    self.binClosingComboBox_2.setCurrentText('Closing')
    self.fillingHolesComboBox_2.setCurrentText('Fill holes')
    self.noiseSpinbox_2.setValue(0)
    self.edgeSpinbox_2.setValue(40)
    self.noise2Spinbox_2.setValue(0)
    self.bin_closingSpinbox_2.setValue(5)
    self.minRegionSpinbox_2.setValue(150)
    self.thresholding_minSpinbox_2.setValue(0)
    self.plotFiltersCheckBox_3.setChecked(False)

    self.dimensionsLineEdit.setToolTip("Image dimensions (in pixels)")
    self.dimensionsLineEdit_2.setToolTip("Image dimensions (in pixels)")

    self.edgeSpinbox.setToolTip(" radius : int "
                                "\n The radius of the disk-shaped neighborhood expressed as a 2-D array of 1’s and 0’s. "
                                "\n The entropy is computed using base 2 logarithm i.e. the filter returns the minimum number "
                                "\n of bits needed to encode the local greylevel distribution.")
    self.edgeSpinbox_2.setToolTip(" radius : int "
                                  "\n The radius of the disk-shaped neighborhood expressed as a 2-D array of 1’s and 0’s. "
                                  "\n The entropy is computed using base 2 logarithm i.e. the filter returns the minimum number "
                                  "\n of bits needed to encode the local greylevel distribution.")

    self.thresholding_minSpinbox.setEnabled(False)
    self.thresholding_minSpinbox_2.setEnabled(False)
    self.noiseSpinbox.setEnabled(False)
    self.noiseSpinbox_2.setEnabled(False)
    self.noise2Spinbox.setEnabled(False)
    self.noise2Spinbox_2.setEnabled(False)

    self.bin_closingSpinbox.setToolTip(' radius : int '
                                       '\n The radius of the disk-shaped neighborhood expressed as a 2-D array of 1’s and 0’s.'
                                       '\n Return fast binary morphological closing of an image.'
                                       '\n Closing can remove small dark spots (i.e.“pepper”) and connect '
                                       '\n small bright cracks. This tends to “close” up (dark) gaps between (bright) features.')

    self.bin_closingSpinbox_2.setToolTip(' radius : int '
                                         '\n The radius of the disk-shaped neighborhood expressed as a 2-D array of 1’s and 0’s.'
                                         '\n Return fast binary morphological closing of an image.'
                                         '\n Closing can remove small dark spots (i.e.“pepper”) and connect '
                                         '\n small bright cracks. This tends to “close” up (dark) gaps between (bright) features.')

    if self.thresholdingComboBox.currentText().split()[1] == '<':
        self.thresholdingComboBox_2.setCurrentText(' '.join([self.thresholdingComboBox.currentText().split()[0], '>']))
    else:
        self.thresholdingComboBox_2.setCurrentText(' '.join([self.thresholdingComboBox.currentText().split()[0], '<']))


def filters_changed(self):
    sender = self.sender()

    if sender.objectName() == 'channelComboBox':

        if self.channelComboBox.currentText() == 'Original':
            self.noise2ComboBox.setCurrentText("None")
            self.noise2ComboBox.setEnabled(False)

            self.thresholdingComboBox.setCurrentText("None")
            self.thresholdingComboBox.setEnabled(False)
            self.thresholding_minSpinbox.setEnabled(False)

            self.edgeComboBox.setCurrentText("Background")
            self.edgeSpinbox.setEnabled(True)
            self.edgeSpinbox.setValue(5)
            self.edgeSpinbox.setToolTip("DBSCAN eps : float"
                                        "\n The maximum distance between two samples for "
                                        "them to be considered as in the same neighborhood.")

    elif sender.objectName() == 'channelComboBox_2':

        if self.channelComboBox_2.currentText() == 'Original':
            self.noise2ComboBox_2.setCurrentText("None")
            self.noise2ComboBox_2.setEnabled(False)

            self.thresholdingComboBox_2.setCurrentText("None")
            self.thresholdingComboBox_2.setEnabled(False)
            self.thresholding_minSpinbox_2.setEnabled(False)

            self.edgeComboBox_2.setCurrentText("Background")
            self.edgeSpinbox_2.setEnabled(True)
            self.edgeSpinbox_2.setValue(20)
            self.edgeSpinbox_2.setToolTip("DBSCAN eps : float"
                                          "\n The maximum distance between two samples for "
                                          "them to be considered as in the same neighborhood.")

    elif sender.objectName() == 'noiseComboBox':

        if self.noiseComboBox.currentText() == 'Gaussian':
            self.noiseSpinbox.setToolTip(" sigma : scalar \n Standard deviation for Gaussian kernel")
            self.noiseSpinbox.setEnabled(True)

        elif self.noiseComboBox.currentText() == 'Median':
            self.noiseSpinbox.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood.")
            self.noiseSpinbox.setEnabled(True)

        elif self.noiseComboBox.currentText() == 'Mean':
            self.noiseSpinbox.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood.")
            self.noiseSpinbox.setEnabled(True)

        else:
            self.noiseSpinbox.setEnabled(False)

    elif sender.objectName() == 'noiseComboBox_2':

        if self.noiseComboBox_2.currentText() == 'Gaussian':
            self.noiseSpinbox_2.setToolTip(
                " sigma : scalar \n Standard deviation for Gaussian kernel. \n Multi-dimensional Gaussian filter.")
            self.noiseSpinbox_2.setEnabled(True)

        elif self.noiseComboBox_2.currentText() == 'Median':
            self.noiseSpinbox_2.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood. \n Return local median of an image.")
            self.noiseSpinbox_2.setEnabled(True)

        elif self.noiseComboBox_2.currentText() == 'Mean':
            self.noiseSpinbox_2.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood. \n Return local mean of an image.")
            self.noiseSpinbox_2.setEnabled(True)

        else:
            self.noiseSpinbox_2.setEnabled(False)

    elif sender.objectName() == 'edgeComboBox':

        if self.edgeComboBox.currentText() == 'Sobel':
            self.edgeSpinbox.setEnabled(False)
            self.edgeSpinbox.setToolTip(" Find the edge magnitude using the Sobel transform.")


        elif self.edgeComboBox.currentText() == 'Canny':
            self.edgeSpinbox.setToolTip(
                " sigma : float \n Standard deviation of the Gaussian filter. \n Edge filter an image using the Canny algorithm.")
            self.edgeSpinbox.setEnabled(True)

        elif self.edgeComboBox.currentText() == 'Entropy':
            self.edgeSpinbox.setToolTip(
                " radius : int \n The radius of the disk-shaped neighborhood expressed as a 2-D array of 1’s and 0’s. \n The entropy is computed using base 2 logarithm i.e. the filter returns the minimum number \n of bits needed to encode the local greylevel distribution.")
            self.edgeSpinbox.setEnabled(True)

        elif self.edgeComboBox.currentText() == 'Background':
            self.edgeSpinbox.setEnabled(True)

        else:
            self.edgeSpinbox.setEnabled(False)

    elif sender.objectName() == 'edgeComboBox_2':

        if self.edgeComboBox_2.currentText() == 'Sobel':
            self.edgeSpinbox_2.setEnabled(False)
            self.edgeSpinbox_2.setToolTip(" Find the edge magnitude using the Sobel transform.")


        elif self.edgeComboBox_2.currentText() == 'Canny':
            self.edgeSpinbox_2.setToolTip(
                " sigma : float \n Standard deviation of the Gaussian filter. \n Edge filter an image using the Canny algorithm.")
            self.edgeSpinbox_2.setEnabled(True)

        elif self.edgeComboBox_2.currentText() == 'Entropy':
            self.edgeSpinbox_2.setToolTip(
                " radius : int \n The radius of the disk-shaped neighborhood expressed as a 2-D array of 1’s and 0’s. \n The entropy is computed using base 2 logarithm i.e. the filter returns the minimum number \n of bits needed to encode the local greylevel distribution.")
            self.edgeSpinbox_2.setEnabled(True)

        elif self.edgeComboBox_2.currentText() == 'Background':
            self.edgeSpinbox_2.setEnabled(True)

        else:
            self.edgeSpinbox_2.setEnabled(False)

    elif sender.objectName() == 'noise2ComboBox':

        if self.noise2ComboBox.currentText() == 'Gaussian':
            self.noise2Spinbox.setToolTip(" sigma : scalar \n Standard deviation for Gaussian kernel")
            self.noise2Spinbox.setEnabled(True)

        elif self.noise2ComboBox.currentText() == 'Median':
            self.noise2Spinbox.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood.")
            self.noise2Spinbox.setEnabled(True)

        elif self.noise2ComboBox.currentText() == 'Mean':
            self.noise2Spinbox.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood.")
            self.noise2Spinbox.setEnabled(True)
        else:
            self.noise2Spinbox.setEnabled(False)

    elif sender.objectName() == 'noise2ComboBox_2':
        if self.noise2ComboBox_2.currentText() == 'Gaussian':
            self.noise2Spinbox_2.setToolTip(
                " sigma : scalar \n Standard deviation for Gaussian kernel. \n Multi-dimensional Gaussian filter.")
            self.noise2Spinbox_2.setEnabled(True)
        elif self.noise2ComboBox_2.currentText() == 'Median':
            self.noise2Spinbox_2.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood. \n Return local median of an image.")
            self.noise2Spinbox_2.setEnabled(True)
        elif self.noise2ComboBox_2.currentText() == 'Mean':
            self.noise2Spinbox_2.setToolTip(
                " radius : int \n The radius of the disk-shaped mask that defines area of \n the image included in the local neighborhood. \n Return local mean of an image.")
            self.noise2Spinbox_2.setEnabled(True)
        else:
            self.noise2Spinbox_2.setEnabled(False)

    elif sender.objectName() == 'thresholdingComboBox':

        if ((self.thresholdingComboBox.currentText() == 'Otsu >') or (
            self.thresholdingComboBox.currentText() == 'Otsu <') or (
                    self.thresholdingComboBox.currentText() == 'Yen >') or (
            self.thresholdingComboBox.currentText() == 'Yen <') or (
                    self.thresholdingComboBox.currentText() == 'Mean >') or (
            self.thresholdingComboBox.currentText() == 'Mean <') or (
                    self.thresholdingComboBox.currentText() == 'Minimum >') or (
            self.thresholdingComboBox.currentText() == 'Minimum <') or (
            self.thresholdingComboBox.currentText() == 'None')):
            self.thresholding_minSpinbox.setEnabled(False)
        else:
            self.thresholding_minSpinbox.setEnabled(True)
            self.thresholding_minSpinbox.setToolTip(
                " multiplier: int \n  Calculation of the background mean and standard deviation. \n  Following of formula: tre = edge_mean - MULTIPLIER * edge_std")

    elif sender.objectName() == 'thresholdingComboBox_2':

        if ((self.thresholdingComboBox_2.currentText() == 'Otsu >') or (
                    self.thresholdingComboBox_2.currentText() == 'Otsu <') or (
                    self.thresholdingComboBox_2.currentText() == 'Yen >') or (
                    self.thresholdingComboBox_2.currentText() == 'Yen <') or (
                    self.thresholdingComboBox_2.currentText() == 'Mean >') or (
                    self.thresholdingComboBox_2.currentText() == 'Mean <') or (
                    self.thresholdingComboBox_2.currentText() == 'Minimum >') or (
                    self.thresholdingComboBox_2.currentText() == 'Minimum <') or (
                    self.thresholdingComboBox_2.currentText() == 'None')):

            self.thresholding_minSpinbox_2.setEnabled(False)
        else:
            self.thresholding_minSpinbox_2.setEnabled(True)
            self.thresholding_minSpinbox_2.setToolTip(
                " multiplier: int \n  Calculation of the background mean and standard deviation. \n  Following of formula: tre = edge_mean - MULTIPLIER * edge_std")

    elif sender.objectName() == 'binClosingComboBox':

        if self.binClosingComboBox.currentText() == 'Closing':
            self.bin_closingSpinbox.setToolTip(
                ' radius : int \n The radius of the disk-shaped the neighborhood expressed as a 2-D array of 1’s and 0’s. \n Return fast binary morphological closing of an image.\n  Closing can remove small dark spots (i.e.“pepper”) and connect \n small bright cracks.This tends to “close” up (dark) gaps between (bright) features.')
            self.bin_closingSpinbox.setEnabled(True)
        else:
            self.bin_closingSpinbox.setEnabled(False)

    elif sender.objectName() == 'binClosingComboBox_2':

        if self.binClosingComboBox_2.currentText() == 'Closing':
            self.bin_closingSpinbox_2.setToolTip(
                ' radius : int \n The radius of the disk-shaped neighborhood expressed as a 2-D array of 1’s and 0’s. \n Return fast binary morphological closing of an image.\n  Closing can remove small dark spots (i.e.“pepper”) and connect \n small bright cracks.This tends to “close” up (dark) gaps between (bright) features.')
            self.bin_closingSpinbox_2.setEnabled(True)
        else:
            self.bin_closingSpinbox_2.setEnabled(False)
