if __name__ == '__main__':
    import sys
    from gui.object_recognition_mainwindow import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    screen = MainWindow()
    screen.resize(600, 320)
    screen.show()

    sys.exit(app.exec_())
