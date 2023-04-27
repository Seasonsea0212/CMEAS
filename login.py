
from PyQt5 import QtCore, QtGui, QtWidgets
import resources_rc

class Ui_MainWindow_login(object):
    def setupUi(self, MainWindow_login):
        MainWindow_login.setObjectName("MainWindow_login")
        MainWindow_login.resize(861, 563)
        MainWindow_login.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        MainWindow_login.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.centralwidget = QtWidgets.QWidget(MainWindow_login)
        self.centralwidget.setObjectName("centralwidget")
        self.left_background = QtWidgets.QLabel(self.centralwidget)
        self.left_background.setGeometry(QtCore.QRect(30, 100, 300, 400))
        font = QtGui.QFont()
        font.setFamily("华文新魏")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.left_background.setFont(font)
        self.left_background.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/images/images/blue_background.jpg);\n"
"border-top-left-radius:10px;\n"
"border-bottom-left-radius:10px;\n"
"")
        self.left_background.setObjectName("left_background")
        self.right_background = QtWidgets.QLabel(self.centralwidget)
        self.right_background.setGeometry(QtCore.QRect(330, 100, 500, 400))
        self.right_background.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-top-right-radius:10px;\n"
"border-bottom-right-radius:10px;")
        self.right_background.setText("")
        self.right_background.setObjectName("right_background")
        self.dirctory = QtWidgets.QWidget(self.centralwidget)
        self.dirctory.setGeometry(QtCore.QRect(490, 130, 174, 41))
        self.dirctory.setObjectName("dirctory")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.dirctory)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_login = QtWidgets.QPushButton(self.dirctory)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_login.setFont(font)
        self.btn_login.setStyleSheet("#btn_login{\n"
"    border:none;\n"
"}\n"
"#btn_login:focus{\n"
"    color: rgb(186, 186, 186);\n"
"}")
        self.btn_login.setObjectName("btn_login")
        self.horizontalLayout.addWidget(self.btn_login)
        self.line = QtWidgets.QFrame(self.dirctory)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.btn_register = QtWidgets.QPushButton(self.dirctory)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_register.setFont(font)
        self.btn_register.setStyleSheet("#btn_register{\n"
"    border:none;\n"
"}\n"
"#btn_register:focus{\n"
"    color:rgb(186,186,186);\n"
"}")
        self.btn_register.setObjectName("btn_register")
        self.horizontalLayout.addWidget(self.btn_register)
        self.widget_login = QtWidgets.QWidget(self.centralwidget)
        self.widget_login.setGeometry(QtCore.QRect(350, 180, 450, 300))
        self.widget_login.setObjectName("widget_login")
        self.lineEdit_un = QtWidgets.QLineEdit(self.widget_login)
        self.lineEdit_un.setGeometry(QtCore.QRect(50, 60, 350, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_un.setFont(font)
        self.lineEdit_un.setStyleSheet("border:1px solid rgb(0,0,0);\n"
"border-radius:5px;\n"
"")
        self.lineEdit_un.setObjectName("lineEdit_un")
        self.lineEdit_pwd = QtWidgets.QLineEdit(self.widget_login)
        self.lineEdit_pwd.setGeometry(QtCore.QRect(50, 120, 350, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_pwd.setFont(font)
        self.lineEdit_pwd.setStyleSheet("border:1px solid rgb(0,0,0);\n"
"border-radius:5px;\n"
"")
        self.lineEdit_pwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_pwd.setObjectName("lineEdit_pwd")
        self.btn_clicked_login = QtWidgets.QPushButton(self.widget_login)
        self.btn_clicked_login.setGeometry(QtCore.QRect(50, 220, 350, 40))
        self.btn_clicked_login.setStyleSheet("#btn_clicked_login{\n"
"    background-color: rgb(43, 128, 255);\n"
"    color:rgb(255, 255, 255);\n"
"    border-color: rgb(43, 128, 255);\n"
"    border-radius:8px;\n"
"}\n"
"#btn_clicked_login:hover{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(43, 128, 255);\n"
"}\n"
"#btn_clicked_login:pressed{\n"
"    padding-top:5px;\n"
"    padding-left:5px;\n"
"}")
        self.btn_clicked_login.setObjectName("btn_clicked_login")
        self.btn_clicked_close = QtWidgets.QPushButton(self.centralwidget)
        self.btn_clicked_close.setGeometry(QtCore.QRect(760, 100, 81, 41))
        self.btn_clicked_close.setStyleSheet("border:none;\n"
"")
        self.btn_clicked_close.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/images/close.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_clicked_close.setIcon(icon)
        self.btn_clicked_close.setObjectName("btn_clicked_close")
        self.widget_register = QtWidgets.QWidget(self.centralwidget)
        self.widget_register.setGeometry(QtCore.QRect(350, 170, 411, 321))
        self.widget_register.setObjectName("widget_register")
        self.lineEdit_input_un = QtWidgets.QLineEdit(self.widget_register)
        self.lineEdit_input_un.setGeometry(QtCore.QRect(50, 30, 350, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_input_un.setFont(font)
        self.lineEdit_input_un.setStyleSheet("border:1px solid rgb(0,0,0);\n"
"border-radius:5px;\n"
"")
        self.lineEdit_input_un.setObjectName("lineEdit_input_un")
        self.lineEdit_input_pwd = QtWidgets.QLineEdit(self.widget_register)
        self.lineEdit_input_pwd.setGeometry(QtCore.QRect(50, 95, 350, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_input_pwd.setFont(font)
        self.lineEdit_input_pwd.setStyleSheet("border:1px solid rgb(0,0,0);\n"
"border-radius:5px;\n"
"")
        self.lineEdit_input_pwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_input_pwd.setObjectName("lineEdit_input_pwd")
        self.btn_clicked_register = QtWidgets.QPushButton(self.widget_register)
        self.btn_clicked_register.setGeometry(QtCore.QRect(50, 235, 350, 40))
        self.btn_clicked_register.setStyleSheet("#btn_clicked_register{\n"
"    background-color: rgb(43, 128, 255);\n"
"    color:rgb(255, 255, 255);\n"
"    border-color: rgb(43, 128, 255);\n"
"    border-radius:8px;\n"
"}\n"
"#btn_clicked_register:hover{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(43, 128, 255);\n"
"}\n"
"#btn_clicked_register:pressed{\n"
"    padding-top:5px;\n"
"    padding-left:5px;\n"
"}")
        self.btn_clicked_register.setObjectName("btn_clicked_register")
        self.lineEdit_sure_pwd = QtWidgets.QLineEdit(self.widget_register)
        self.lineEdit_sure_pwd.setGeometry(QtCore.QRect(50, 155, 350, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_sure_pwd.setFont(font)
        self.lineEdit_sure_pwd.setStyleSheet("border:1px solid rgb(0,0,0);\n"
"border-radius:5px;\n"
"")
        self.lineEdit_sure_pwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_sure_pwd.setObjectName("lineEdit_sure_pwd")
        self.btn_admin_login = QtWidgets.QPushButton(self.centralwidget)
        self.btn_admin_login.setGeometry(QtCore.QRect(340, 110, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.btn_admin_login.setFont(font)
        self.btn_admin_login.setStyleSheet("#btn_admin_login{\n"
"    border:none;\n"
"}\n"
"#btn_admin_login:focus{\n"
"    color: rgb(186, 186, 186);\n"
"}")
        self.btn_admin_login.setObjectName("btn_admin_login")
        MainWindow_login.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow_login)
        self.btn_clicked_close.clicked.connect(MainWindow_login.close)
        self.btn_admin_login.clicked.connect(MainWindow_login.open_win_adminLogin)
        QtCore.QMetaObject.connectSlotsByName(MainWindow_login)

    def retranslateUi(self, MainWindow_login):
        _translate = QtCore.QCoreApplication.translate
        MainWindow_login.setWindowTitle(_translate("MainWindow_login", "MainWindow"))
        self.left_background.setText(_translate("MainWindow_login", "<html>\n"
"<head/>\n"
"<body>\n"
"<p align=\"center\">\n"
"<span style=\" font-size:48pt; color:#ffffff;\">中文</span></p>\n"
"<p align=\"center\">\n"
"<span style=\" font-size:48pt; color:#ffffff;\">微情绪</span></p>\n"
"<p align=\"center\">\n"
"<span style=\" font-size:48pt; color:#ffffff;\">分析系统</span></p>\n"
"</body>\n"
"</html>"))
        self.btn_login.setText(_translate("MainWindow_login", "登录"))
        self.btn_register.setText(_translate("MainWindow_login", "注册"))
        self.lineEdit_un.setPlaceholderText(_translate("MainWindow_login", "用户名："))
        self.lineEdit_pwd.setPlaceholderText(_translate("MainWindow_login", "密码："))
        self.btn_clicked_login.setText(_translate("MainWindow_login", "登录"))
        self.lineEdit_input_un.setPlaceholderText(_translate("MainWindow_login", "输入用户名："))
        self.lineEdit_input_pwd.setPlaceholderText(_translate("MainWindow_login", "设置密码："))
        self.btn_clicked_register.setText(_translate("MainWindow_login", "注册"))
        self.lineEdit_sure_pwd.setPlaceholderText(_translate("MainWindow_login", "确认密码："))
        self.btn_admin_login.setText(_translate("MainWindow_login", "管理员登录"))

