# How to beat human in damage recognition?

Bachelor thesis project "AUTOMATED MORPHOLOGICAL ANALYSIS OF LASER-INDUCED DAMAGE:
INVESTIGATION OF FATIGUE EFFECT IN ZRO2 OPTICAL COATING" at Vilnius University, 2018.

Thesis presentation details how the content of this repository can be used for laser-induced damage detection and classification, and can be assessed [here](https://docs.google.com/presentation/d/1SBlaLdjNwfMvd1MDDXGLrlrlZgE0Cdfuq1SQTEXR7vo/edit?usp=sharing) (in lithuanian).

** Note: no examples are provided which are required for the applications. **

## Directory content
* image_processing - application for feature extraction from LIDT images.
* image_classification  - application for classification of extracted features from LIDT images.

## Directory tree
```bash
image_classification
   |-- __init__.py
   |-- gui
   |   |-- cluster_groupBox.py
   |   |-- features_dialog2.py
   |   |-- features_dialog2_ui.py
   |   |-- features_dialog2_ui.ui
   |   |-- info_clustering_dialog.py
   |   |-- info_clustering_dialog_ui.py
   |   |-- info_clustering_dialog_ui.ui
   |   |-- object_recognition_mainwindow.py
   |   |-- object_recognition_mainwindow_ui.py
   |   |-- object_recognition_mainwindow_ui.ui
   |   |-- outliers_dialog.py
   |   |-- outliers_dialog_ui.py
   |   |-- outliers_dialog_ui.ui
   |   |-- outliers_groupBox.py
   |   |-- radius_dialog.py
   |   |-- radius_dialog_ui.py
   |   |-- radius_dialog_ui.ui
   |-- logic
   |   |-- recognition.py
   |-- main.py
image_processing
   |-- __init__.py
   |-- gui
   |   |-- features_dialog.py
   |   |-- features_dialog2_ui.py
   |   |-- features_dialog_ui.py
   |   |-- features_dialog_ui.ui
   |   |-- mainwindow.py
   |   |-- mainwindow_ui.py
   |   |-- mainwindow_ui.ui
   |   |-- set_tips_and_buttons.py
   |-- logic
   |   |-- identification.py
   |-- main.py
LICENSE
README.md
cv_372.yml
```
## Usage
#### Create Anaconda environment
Use the conda environment file **cv_372.yml** to install the required **cv_372** environment and its modules.

```bash
conda env create -f cv_372.yml
```
Activate `cv_372` conda environment:
```bash
conda activate experiments
```

#### Run one of the applications
Navigate to one of the applications and run `main.py` files:
```bash
cd image_processing
python main.py
```
