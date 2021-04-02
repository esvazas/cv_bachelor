from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from gui.radius_dialog_ui import Ui_Dialog

class GraphCanvas(FigureCanvas):

    def __init__(self, height, width, limit, min_limit, radius = 150):
        self.fig = Figure(figsize=(8,8),dpi=60)

        self.height2 = height
        self.width2 = width
        self.limit = limit
        self.min_limit = min_limit

        self.draw_graph(radius)
        FigureCanvas.__init__(self, self.fig)

    def draw_graph(self, radius):

        self.ax0 = self.fig.add_subplot(111)

        # Eliminate upper and right axes
        self.ax0.spines['top'].set_visible(False)
        self.ax0.spines['right'].set_visible(False)

        # Show ticks on the left and lower axes only
        self.ax0.xaxis.set_tick_params(bottom='on', top='off')
        self.ax0.yaxis.set_tick_params(left='on', right='off')

        self.ax0.set_xlim(-np.round((self.width2/self.limit),1), np.round((self.width2/self.limit),1))
        self.ax0.set_ylim(-np.round((self.height2/self.limit),1), np.round((self.height2/self.limit),1))

        # Set the axis ticks
        self.ax0.set_xticks([-np.round((self.width2/self.limit),1),-np.round((self.width2/self.limit),1)/2,
                             np.round((self.width2/self.limit),1)/2, np.round((self.width2/self.limit),1)])

        self.ax0.set_yticks([-np.round((self.height2/self.limit),1),-np.round((self.height2/self.limit),1)/2,
                              np.round((self.height2/self.limit),1)/2,np.round((self.height2/self.limit),1)])

        # Move remaining spines to the center
        self.ax0.spines['bottom'].set_position('center') # spine for xaxis
        self.ax0.spines['left'].set_position('center')  # spine for yaxis

        self.draw_circle(radius)


    def draw_circle(self, radius):
        print("Radius",radius/(self.min_limit))
        circle1 = plt.Circle((0, 0), radius/(self.min_limit), color='r')
        self.ax0.add_artist(circle1)


class ShowGraph(QDialog, Ui_Dialog):
    def __init__(self, shape, radius=0.5):
        super().__init__()
        self.radius = radius
        self.height2, self.width2, _ = shape

        # The graph's shape should represent the actual shape of the pictures
        # Find the max element -> devide all axis by it.

        self.limit, self.min_limit = max([self.height2,self.width2]), min([self.height2,self.width2])


        self.setupUi()

    def setupUi(self):
        super().setupUi(self)

        self.graph =  GraphCanvas(self.height2, self.width2, self.limit, self.min_limit, self.radius,)
        self.horizontalLayout_2.addWidget(self.graph)

        self.radiusHorizontalSlider.setMinimum(0)
        self.radiusHorizontalSlider.setMaximum(self.min_limit/2)
        self.radiusDoubleSpinBox.setMinimum(0)
        self.radiusDoubleSpinBox.setMaximum(self.min_limit/2)

        self.radiusHorizontalSlider.valueChanged.connect(self.update_radiusValue_slider)
        self.radiusDoubleSpinBox.valueChanged.connect(self.update_radiusValue_spinBox)

    def update_radiusValue_spinBox(self):

        self.radiusHorizontalSlider.setValue(self.radiusDoubleSpinBox.value())
        self.radius = self.radiusHorizontalSlider.value()

        self.graph.draw_graph(self.radius)


    def update_radiusValue_slider(self):

        self.radiusDoubleSpinBox.setValue(self.radiusHorizontalSlider.value())
        self.radius = self.radiusDoubleSpinBox.value()

        self.graph.draw_graph(self.radius)






        print("Cia")
        print(self.radiusHorizontalSlider.value())
        # Set the zero point in the axis interaction point

