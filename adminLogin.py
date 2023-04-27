
from PyQt5 import QtCore, QtGui, QtWidgets
import resources_rc

class Ui_MainWindow_adminLogin(object):
    def setupUi(self, MainWindow_adminLogin):
        MainWindow_adminLogin.setObjectName("MainWindow_adminLogin")
        MainWindow_adminLogin.resize(705, 467)
        MainWindow_adminLogin.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        MainWindow_adminLogin.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.centralwidget = QtWidgets.QWidget(MainWindow_adminLogin)
        self.centralwidget.setObjectName("centralwidget")
        self.widget_adminLogin = QtWidgets.QWidget(self.centralwidget)
        self.widget_adminLogin.setGeometry(QtCore.QRect(240, 100, 231, 231))
        self.widget_adminLogin.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-top-right-radius:10px;\n"
"border-bottom-right-radius:10px;\n"
"border-top-left-radius:10px;\n"
"border-bottom-left-radius:10px;")
        self.widget_adminLogin.setObjectName("widget_adminLogin")
        self.label_title_adminLogin = QtWidgets.QLabel(self.widget_adminLogin)
        self.label_title_adminLogin.setGeometry(QtCore.QRect(-20, 10, 250, 70))
        self.label_title_adminLogin.setStyleSheet("font: 18pt \"方正姚体\";\n"
"")
        self.label_title_adminLogin.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title_adminLogin.setObjectName("label_title_adminLogin")
        self.btn_clicked_close_2 = QtWidgets.QPushButton(self.widget_adminLogin)
        self.btn_clicked_close_2.setGeometry(QtCore.QRect(180, 0, 51, 31))
        self.btn_clicked_close_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-top-right-radius:10px;\n"
"border-bottom-right-radius:10px;\n"
"border-top-left-radius:10px;\n"
"border-bottom-left-radius:10px;\n"
"")
        self.btn_clicked_close_2.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/images/close.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_clicked_close_2.setIcon(icon)
        self.btn_clicked_close_2.setObjectName("btn_clicked_close_2")
        self.widget_adminLogin_2 = QtWidgets.QWidget(self.widget_adminLogin)
        self.widget_adminLogin_2.setGeometry(QtCore.QRect(-20, 70, 250, 150))
        self.widget_adminLogin_2.setObjectName("widget_adminLogin_2")
        self.btn_clicked_login_2 = QtWidgets.QPushButton(self.widget_adminLogin_2)
        self.btn_clicked_login_2.setGeometry(QtCore.QRect(50, 90, 150, 40))
        self.btn_clicked_login_2.setStyleSheet("#btn_clicked_login_2{\n"
"background-color: rgb(43, 128, 255);\n"
"color:rgb(255, 255, 255);\n"
"border-color: rgb(43, 128, 255);\n"
"border-radius:8px;\n"
"}\n"
"#btn_clicked_login_2:hover{\n"
"background-color: rgb(255, 255, 255);\n"
"color: rgb(43, 128, 255);\n"
"}\n"
"#btn_clicked_login_2:pressed{\n"
"padding-top:5px;\n"
"padding-left:5px;\n"
"}")
        self.btn_clicked_login_2.setObjectName("btn_clicked_login_2")
        self.lineEdit_pwd_adminLogin = QtWidgets.QLineEdit(self.widget_adminLogin_2)
        self.lineEdit_pwd_adminLogin.setGeometry(QtCore.QRect(30, 30, 200, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_pwd_adminLogin.setFont(font)
        self.lineEdit_pwd_adminLogin.setStyleSheet("border:1px solid rgb(0,0,0);\n"
"border-radius:5px;\n"
"")
        self.lineEdit_pwd_adminLogin.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_pwd_adminLogin.setObjectName("lineEdit_pwd_adminLogin")
        MainWindow_adminLogin.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow_adminLogin)
        self.btn_clicked_close_2.clicked.connect(MainWindow_adminLogin.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow_adminLogin)

    def retranslateUi(self, MainWindow_adminLogin):
        _translate = QtCore.QCoreApplication.translate
        MainWindow_adminLogin.setWindowTitle(_translate("MainWindow_adminLogin", "MainWindow"))
        self.label_title_adminLogin.setText(_translate("MainWindow_adminLogin", "管理员登录"))
        self.btn_clicked_login_2.setText(_translate("MainWindow_adminLogin", "登录"))
        self.lineEdit_pwd_adminLogin.setPlaceholderText(_translate("MainWindow_adminLogin", "密码："))

