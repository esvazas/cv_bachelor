if __name__ == '__main__':
    import sys
    from gui.mainwindow import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    screen = MainWindow()
    screen.resize(1100, 700)
    screen.show()

    sys.exit(app.exec_())
