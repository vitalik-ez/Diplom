from PyQt5 import QtWidgets
import sys
from view import View
from controller import Controller
from model import Model


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    view = View()
    view.show()

    model = Model()

    Controller(view=view, model=model)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()