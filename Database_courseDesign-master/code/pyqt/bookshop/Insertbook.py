# -*- coding: utf-8 -*-

import sys
from pymysql import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication


class Ui_InputBookinfo(object):
    def execute_sql(self, sql):
        try:
            # 执行数据库操作
            self.cursor.execute(sql)
            # 事务提交
            self.db.commit()
        except Exception as err:
            # 事务回滚
            self.db.rollback()
            print("SQL执行错误，原因：", err)

    def setupUi(self, InputBookinfo):
        InputBookinfo.setObjectName("InputBookinfo")
        InputBookinfo.resize(516, 326)
        self.pushButton_2 = QtWidgets.QPushButton(InputBookinfo)
        self.pushButton_2.setGeometry(QtCore.QRect(400, 64, 81, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.widget = QtWidgets.QWidget(InputBookinfo)
        self.widget.setGeometry(QtCore.QRect(80, 230, 301, 31))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.widget1 = QtWidgets.QWidget(InputBookinfo)
        self.widget1.setGeometry(QtCore.QRect(90, 60, 291, 151))
        self.widget1.setObjectName("widget1")
        self.gridLayout = QtWidgets.QGridLayout(self.widget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.widget1)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget1)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget1)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget1)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 1, 1, 1)

        self.statusbar = QtWidgets.QStatusBar(InputBookinfo)
        self.statusbar.setObjectName("statusbar")
        InputBookinfo.setStatusBar(self.statusbar)

        self.retranslateUi(InputBookinfo)
        self.pushButton_2.clicked.connect(self.lineEdit_4.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_3.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_2.clear)
        self.pushButton_2.clicked.connect(self.lineEdit.clear)
        self.pushButton_3.clicked.connect(InputBookinfo.close)
        QtCore.QMetaObject.connectSlotsByName(InputBookinfo)

        self.pushButton.clicked.connect(lambda: self.get_book_info())

    def retranslateUi(self, InputBookinfo):
        _translate = QtCore.QCoreApplication.translate
        InputBookinfo.setWindowTitle(_translate("InputBookinfo", "添加书籍信息"))
        self.pushButton_2.setText(_translate("InputBookinfo", "清空"))
        self.pushButton.setText(_translate("InputBookinfo", "确认"))
        self.pushButton_3.setText(_translate("InputBookinfo", "退出"))
        self.label.setText(_translate("InputBookinfo", "输入ISBN"))
        self.label_2.setText(_translate("InputBookinfo", "书名"))
        self.label_3.setText(_translate("InputBookinfo", "作者"))
        self.label_4.setText(_translate("InputBookinfo", "价格"))

    def get_book_info(self):
        self.db = connect(host='localhost', port=5432, charset='utf8', database='Bookshop_Manage_system', password='root',
                          user='postgres')
        # 创建游标对象
        self.cursor = self.db.cursor()
        sql = "use Bookshop_Manage_system"
        self.cursor.execute(sql)
        ISBN = self.lineEdit.text()  # 获取文本框内容
        bookName = self.lineEdit_2.text()
        author = self.lineEdit_3.text()  # 获取文本框内容
        price = self.lineEdit_4.text()
        sql = "SELECT * FROM book"
        self.cursor.execute(sql)
        # 获取查询到的数据, 是以二维元组的形式存储的, 所以读取需要使用 data[i][j] 下标定位
        data = self.cursor.fetchall()
        original_row = len(data)
        # 定义SQL语句
        try:
            sql = "insert into book(ISBN, bookname, author, price) values('%s','%s','%s','%d')" % (
                ISBN, bookName, author, int(price))
            self.execute_sql(sql)
        except Exception as err:
            print("错误原因 " + err)
        sql = "SELECT * FROM book"
        self.cursor.execute(sql)
        # 获取查询到的数据, 是以二维元组的形式存储的, 所以读取需要使用 data[i][j] 下标定位
        data = self.cursor.fetchall()
        changed_row = len(data)
        if changed_row == original_row+1:
             self.statusbar.showMessage("插入成功，请返回刷新", 2000)
        else:
            self.statusbar.showMessage("插入失败，请重试", 2000)
        self.cursor.close()


if __name__ == '__main__':
    # db = QtSql.QSqlDatabase.addDatabase('QMYSQL')
    # db.setPort(3306)
    # db.setHostName('localhost')
    # db.setDatabaseName('bookshopmanagement')
    # db.setUserName('root')
    # db.setPassword('zyh20000205')

    # print(db.open())

    app = QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    test_ui = Ui_InputBookinfo()
    test_ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())