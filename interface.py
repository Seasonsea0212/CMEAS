
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QMovie
from PyQt5.QtWidgets import QMessageBox


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(818, 441)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 821, 461))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(0, 100, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(310, 330, 100, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.textEdit = QtWidgets.QTextEdit(self.tab)
        self.textEdit.setGeometry(QtCore.QRect(110, 20, 681, 211))
        self.textEdit.setObjectName("textEdit")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_5.setGeometry(QtCore.QRect(420, 330, 100, 41))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(330, 260, 191, 51))
        self.pushButton.setObjectName("pushButton")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setGeometry(QtCore.QRect(0, 30, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        font = QtGui.QFont()
        font.setPointSize(13)
        self.textEdit.setFont(font)
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_2.setGeometry(QtCore.QRect(160, 20, 611, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_2.setGeometry(QtCore.QRect(160, 60, 611, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(0, 140, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setGeometry(QtCore.QRect(160, 110, 141, 31))
        self.label_6.setObjectName("label_6")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_3.setGeometry(QtCore.QRect(290, 110, 481, 31))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(160, 150, 611, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_4.setGeometry(QtCore.QRect(160, 190, 611, 31))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_4.setGeometry(QtCore.QRect(250, 250, 271, 51))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(380, 350, 151, 21))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(320, 340, 61, 41))
        self.label_8.setObjectName("label_8")
        self.tabWidget.addTab(self.tab_2, "")

        self.gif = QMovie('images/loading.gif')
        self.label_8.setMovie(self.gif)
        self.label_8.setScaledContents(True)
        self.gif.start()

        # 添加功能函数
        self.pushButton_2.clicked.connect(self.uploadfile)
        self.pushButton_3.clicked.connect(self.choosefile)
        self.tabWidget.setStyleSheet("pane{top:-1px;};")
        self.tabWidget.setStyleSheet("background-image: url(./images/background.png);")  # 设置背景图片
        Form.setStyleSheet("background-color:#E1FFFF;")  # 设置背景颜色
        # self.tabWidget.setStyleSheet("background-color:#C8C8A9;")
        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "中文微情感分析系统"))
        Form.setWindowIcon(QIcon('images//logo.ico'))
        # self.label.setText(_translate("Form", "要分析的文本"))
        self.label_2.setText(_translate("Form", "文本内容"))
        self.label_3.setText(_translate("Form", "情感类别："))
        self.textEdit.setPlaceholderText(_translate("Form", "请输入要进行情感分析的文本"))
        self.lineEdit_5.setPlaceholderText(_translate("Form", "情感类别"))
        self.pushButton.setText(_translate("Form", "单条文本情感分类"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "单条文本情感分类"))
        self.label_4.setText(_translate("Form", "批量导入文本"))
        self.pushButton_2.setText(_translate("Form", "选择Excel文件"))
        self.lineEdit_2.setPlaceholderText(_translate("Form", "文件路径"))
        self.label_5.setText(_translate("Form", "设置输出路径"))
        self.label_6.setText(_translate("Form", "输出文件名称"))
        self.lineEdit_3.setPlaceholderText(_translate("Form", "请输入输出文件名称"))
        self.pushButton_3.setText(_translate("Form", "选择输出文件夹"))
        self.lineEdit_4.setPlaceholderText(_translate("Form", "输出文件路径"))
        self.pushButton_4.setText(_translate("Form", "进行批量文本情感分类"))
        self.label_7.setText(_translate("Form", "分类中···"))
        self.label_2.setStyleSheet("background: transparent;")
        self.label_3.setStyleSheet("background: transparent;")
        self.label_4.setStyleSheet("background: transparent;")
        self.label_5.setStyleSheet("background: transparent;")
        self.label_6.setStyleSheet("background: transparent;")
        self.label_7.setStyleSheet("background: transparent;")
        self.label_8.setStyleSheet("background: transparent;")
        self.pushButton_2.setStyleSheet(
            '''QPushButton{background:#AFEEEE;border-radius:5px;}QPushButton:hover{background:#00FFFF;}''')
        self.pushButton_3.setStyleSheet(
            '''QPushButton{background:#AFEEEE;border-radius:5px;}QPushButton:hover{background:#00FFFF;}''')
        self.pushButton.setStyleSheet(
            '''QPushButton{background:#AFEEEE;border-radius:15px;}QPushButton:hover{background:#00FFFF;}''')
        self.pushButton_4.setStyleSheet(
            '''QPushButton{background:#AFEEEE;border-radius:15px;}QPushButton:hover{background:#00FFFF;}''')
        # self.label_8.setText(_translate("Form", "logo"))
        # 设置控件隐藏
        self.label_3.setVisible(False)
        self.lineEdit_5.setVisible(False)
        self.label_7.setVisible(False)
        self.label_8.setVisible(False)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "批量文本情感分类"))

    # 添加功能函数：
    # 选择excel文件
    def uploadfile(self, Filepath):
        x, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选取excel文件", "./", "*.xlsx")
        self.lineEdit_2.setText(x)

    # 设置输出路径
    def choosefile(self, Filepath):
        # 选择输出路径
        f = QtWidgets.QFileDialog.getExistingDirectory(None, "选取输出文件夹", "./")  # 起始路径
        # 输出文件名称
        filename = self.lineEdit_3.text()
        if filename.endswith('.xlsx'):
            filename = filename[:-5]
        else:
            filename = filename
        if filename == '':
            QMessageBox().warning(None, "警告", "请先补全输出文件名称！", QMessageBox.Close)
        else:
            f += '/'
            f += filename
            f += '.xlsx'   # 后续此处需要添加支持其他的格式
            self.lineEdit_4.setText(f)

    # 显示加载中动态效果
    def showloading(self):
        self.label_7.setVisible(True)
        self.label_8.setVisible(True)

    # 隐藏加载中动态效果
    def cancelloading(self):
        self.label_7.setVisible(False)
        self.label_8.setVisible(False)

    # 警告提示
    def warn1(self):
        QMessageBox().warning(None, "警告", "文本内容不能为空！", QMessageBox.Close)

    def warn2(self):
        QMessageBox().warning(None, "警告", "未选择要进行批量分类的文本！", QMessageBox.Close)

    def warn3(self):
        QMessageBox().warning(None, "警告", "未设置结果文件输出路径！", QMessageBox.Close)

    def success(self):
        QMessageBox().information(None, "提示", "批量情感分类完成！", QMessageBox.Close)