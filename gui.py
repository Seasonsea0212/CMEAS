import paddlehub as hub
import sys
import xlrd
import csv
import re
import pandas as pd
import numpy as np
from functools import partial
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from time import time

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from CMSAS.admin import Ui_MainWindow_admin
from CMSAS.adminLogin import Ui_MainWindow_adminLogin
from CMSAS.interface import Ui_Form
from CMSAS.login import Ui_MainWindow_login
from CMSAS.connect import connect

# 链接数据库
cursor, conn = connect()

# 管理员登录界面
class AdminLogin_ui(QMainWindow, Ui_MainWindow_adminLogin):
    def __init__(self, parent=None):
        super(AdminLogin_ui, self).__init__(parent)
        self.setupUi(self)

        # “管理员登录”按钮：点击进入管理员登录界面，成功登录后进入后台管理界面
        self.btn_clicked_login_2.clicked.connect(self.adminLogin)

    # 管理员登录界面功能
    # 一、管理员登录
    def adminLogin(self):
        adminLogin_pwd = self.lineEdit_pwd_adminLogin.text()
        if adminLogin_pwd == "":
            QMessageBox.warning(self, "警告", "请输入管理员密码", QMessageBox.Yes)
        elif adminLogin_pwd != "root":
            QMessageBox.warning(self, "警告", "管理员密码错误！请重新输入", QMessageBox.Yes)
        else:
            self.open_win_admin()
            self.close()
            win.hide()
    # 打开后台管理界面
    def open_win_admin(self):
        win_admin.show()
    # 实现窗口拖动
    def mousePressEvent(self, event):  # 重写鼠标点击事件
        if event.button() == QtCore.Qt.LeftButton:
            self.Move = True  # 鼠标按下时设置为移动状态
            self.Point = event.globalPos() - self.pos()  # 记录起始点坐标
            event.accept()
    def mouseMoveEvent(self, QMouseEvent):  # 移动时间
        if QtCore.Qt.LeftButton and self.Move:  # 切记这里的条件不能写死，只要判断move和鼠标执行即可
            self.move(QMouseEvent.globalPos() - self.Point)  # 移动到鼠标到达的坐标点
            QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):  # 结束事件
        self.Move = False


# 后台管理界面
class Admin_ui(QMainWindow, Ui_MainWindow_admin):
    def __init__(self, parent=None):
        super(Admin_ui, self).__init__(parent)
        self.setupUi(self)

        # “返回”按钮：点击即返回登录界面，再次进入需管理员重新登录
        self.btn_admin_return.clicked.connect(self.returnTo_win_login)

    # 退出后台系统，返回登录界面
    def returnTo_win_login(self):
        win_admin.close()
        win.show()
        win_adminLogin.lineEdit_pwd_adminLogin.setText("")
        win_admin.tableWidget_1.clear()
    # 实现窗口拖动
    def mousePressEvent(self, event):  # 重写鼠标点击事件
        if event.button() == QtCore.Qt.LeftButton:
            self.Move = True  # 鼠标按下时设置为移动状态
            self.Point = event.globalPos() - self.pos()  # 记录起始点坐标
            event.accept()
    def mouseMoveEvent(self, QMouseEvent):  # 移动时间
        if QtCore.Qt.LeftButton and self.Move:  # 切记这里的条件不能写死，只要判断move和鼠标执行即可
            self.move(QMouseEvent.globalPos() - self.Point)  # 移动到鼠标到达的坐标点
            QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):  # 结束事件
        self.Move = False


# 程序首页（登录界面）
class MyMainForm(QMainWindow, Ui_MainWindow_login):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.widget_register.hide()

        # 切换登录、注册界面
        self.btn_login.clicked.connect(self.changeTo_widget_login)
        self.btn_register.clicked.connect(self.changeTo_widget_register)
        # 打开管理员登录窗口
        self.btn_admin_login.clicked.connect(self.open_win_adminLogin)

        # 登录，进入系统
        self.btn_clicked_login.clicked.connect(self.login)
        # 注册，将设置的用户名、密码存入数据库
        self.btn_clicked_register.clicked.connect(self.register)

    # 登录界面功能
    # 一、用户登录
    def login(self):
        un = self.lineEdit_un.text()
        pwd = self.lineEdit_pwd.text()
        if un == '' or pwd == '':
            QMessageBox.warning(self, "警告", "请输入用户名/密码", QMessageBox.Yes)
        else:
            sql = 'select * from user where username = "%s" and password = "%s"' % (un, pwd)
            res = cursor.execute(sql)
            if res:
                win_system.show()
                QMessageBox.warning(self, "提示", "登录成功！进入情感分析系统", QMessageBox.Yes)
                self.close()
                pass
            else:
                QMessageBox.warning(self, "警告", "密码错误，请重新输入！", QMessageBox.Yes)

    # 二、用户注册
    def register(self):
        un = self.lineEdit_input_un.text()
        pwd1 = self.lineEdit_input_pwd.text()
        pwd2 = self.lineEdit_sure_pwd.text()
        if un == '' or pwd1 == '':
            QMessageBox.warning(self, "警告", "请设置用户名/密码", QMessageBox.Yes)
        elif pwd2 == '':
            QMessageBox.warning(self, "警告", "请确认密码", QMessageBox.Yes)
        elif pwd1 != pwd2:
            QMessageBox.warning(self, "警告", "与设置密码不同！请重新输入", QMessageBox.Yes)
        else:
            sql = 'SELECT * FROM user WHERE username = "%s"' % un
            res = cursor.execute(sql)
            if res:
                QMessageBox.warning(self, "警告", "该用户名已存在！", QMessageBox.Yes)
            else:
                value = (un, pwd1)
                sql = 'INSERT INTO user(`username`, `password`) VALUES ("%s", "%s");' % value
                cursor.execute(sql)
                conn.commit()
                QMessageBox.warning(self, "提示", "注册成功！请返回登录页登录", QMessageBox.Yes)

    # 登录/注册功能界面切换方法
    def changeTo_widget_register(self):
        self.widget_login.hide()
        self.widget_register.show()
    def changeTo_widget_login(self):
        self.widget_register.hide()
        self.widget_login.show()
    # 打开管理员登录界面方法
    def open_win_adminLogin(self):
        win_adminLogin.show()
    # 实现窗口拖动
    def mousePressEvent(self, event):  # 重写鼠标点击事件
        if event.button() == QtCore.Qt.LeftButton:
            self.Move = True  # 鼠标按下时设置为移动状态
            self.Point = event.globalPos() - self.pos()  # 记录起始点坐标
            event.accept()
    def mouseMoveEvent(self, QMouseEvent):  # 移动时间
        if QtCore.Qt.LeftButton and self.Move:  # 切记这里的条件不能写死，只要判断move和鼠标执行即可
            self.move(QMouseEvent.globalPos() - self.Point)  # 移动到鼠标到达的坐标点
            QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):  # 结束事件
        self.Move = False


# 系统界面
class Form_ui(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(Form_ui, self).__init__(parent)
        self.setupUi(self)

        # 为按钮绑定相关功能函数完成功能添加：
        # 单条文本情感分类
        self.pushButton.clicked.connect(partial(Single_classification, self))
        # 批量文本情感分类
        self.pushButton_4.clicked.connect(partial(Batch_classification, self))

    # 实现窗口拖动
    def mousePressEvent(self, event):  # 重写鼠标点击事件
        if event.button() == QtCore.Qt.LeftButton:
            self.Move = True  # 鼠标按下时设置为移动状态
            self.Point = event.globalPos() - self.pos()  # 记录起始点坐标
            event.accept()
    def mouseMoveEvent(self, QMouseEvent):  # 移动时间
        if QtCore.Qt.LeftButton and self.Move:  # 切记这里的条件不能写死，只要判断move和鼠标执行即可
            self.move(QMouseEvent.globalPos() - self.Point)  # 移动到鼠标到达的坐标点
            QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):  # 结束事件
        self.Move = False

# 单条文本情感分类
def Single_classification(ui):
    content = ui.textEdit.toPlainText()  # 获取输入的要进行情感分类的文本
    # 要进行情感分类的文本内容不能为空
    if content == '':
        ui.label_3.setVisible(False)     # 隐藏结果
        ui.lineEdit_5.setVisible(False)
        ui.warn1()   # 提示补全文本内容
    else:
        # 格式处理：
        data = []
        list = []
        list.append(content)
        data.append(list)
        t1 = time()
        # 对单条文本进行预测
        label = model.predict(data, max_seq_len=128, batch_size=16, use_gpu=False)  # 若下载了GPU的paddle，可以将此处use_gpu设置为True
        t2 = time()
        # 单条预测时间检测
        print('单条文本分类CPU环境下预测耗时（毫秒）：%.3f' % ((t2 - t1) * 1000.0))
        ui.lineEdit_5.setText(label[0])   # 完成预测后在界面显示预测的情感类别
        ui.label_3.setVisible(True)
        ui.lineEdit_5.setVisible(True)

# 批量文本情感分类
def Batch_classification(win_system):
    excel_path = win_system.lineEdit_2.text()   # 获取输入文件路径
    output_path = win_system.lineEdit_4.text()  # 获取输出文件路径
    # 路径不能为空
    if excel_path == '':
        win_system.warn2()  # 提示未选择要进行批量情感分类的excel文件
    elif output_path == '':
        win_system.warn3()  # 提示未选择生成结果文件输出路径
    else:
        # ui.showloading()   # 显示加载中
        # 读取导入的excel文件
        df = pd.read_excel(excel_path)
        # 格式处理：
        news = pd.DataFrame(columns=['content'])
        news['content'] = df["content"]
        # 首先将pandas读取的数据转化为array
        data_array = np.array(news)
        # 然后转化为list形式
        data_list =data_array.tolist()

        # 批量文本预测
        results = model.predict(data_list, max_seq_len=128, batch_size=16, use_gpu=False) # 若下载了GPU的paddle，可以将此处use_gpu设置为True

        df['label'] = results # 将结果填充到label列上
        # 保存结果文件为excel文件
        df.to_excel(output_path, sheet_name='预测结果', index=False, header=True)
        # ui.cancelloading() # 完成预测后取消显示加载中
        win_system.success()  # 提示分类完成

if __name__ == '__main__':

    # 定义要进行情感分类的7个类别
    label_list=['难过', '愉快', '喜欢', '愤怒', '害怕', '惊讶', '厌恶']
    label_map = { 
        idx: label_text for idx, label_text in enumerate(label_list)
    }

    # 加载训练好的模型
    model = hub.Module(
        name='ernie_tiny',
        version='2.0.2',  # 若未指定版本将自动下载最新的版本
        task='seq-cls',
        num_classes=7,
        load_checkpoint='./best_model/model.pdparams',
        label_map=label_map
    )

    app = QApplication(sys.argv)
    # 初始化
    win = MyMainForm()
    win_adminLogin = AdminLogin_ui()
    win_admin = Admin_ui()
    win_system = Form_ui()
    # 将窗口控件显示在屏幕上
    win.show()

    sys.exit(app.exec_())
    input('Press Enter to exit...')