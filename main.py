import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from gui import MainFrame

def run():
    app = QtWidgets.QApplication([])
    app.setApplicationDisplayName('Keras Training Framework')

    screen_res = app.desktop().screenGeometry()
    screen_width, screen_height = screen_res.width(), screen_res.height()
    window = MainFrame(screen_width//4, screen_height//3)
    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    run()