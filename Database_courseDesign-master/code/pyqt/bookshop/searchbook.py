# -*- coding: utf-8 -*-

import sys

from PyQt5 import QtCore, QtGui, QtWidgets,QtSql
from PyQt5.QtWidgets import QApplication
from pymysql import *

class Ui_searchbook(object):
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

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(726, 491)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(510, 70, 93, 21))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(510, 140, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.tableView = QtWidgets.QTableView(Form)
        self.tableView.setGeometry(QtCore.QRect(110, 210, 491, 211))
        self.tableView.setObjectName("tableView")
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(110, 60, 341, 111))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.statusbar = QtWidgets.QStatusBar(Form)
        self.statusbar.setObjectName("statusbar")
        Form.setStatusBar(self.statusbar)
        self.retranslateUi(Form)
        self.pushButton_2.clicked.connect(Form.close)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.pushButton.clicked.connect(lambda: self.searchbook())
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "搜索书籍"))
        self.pushButton.setText(_translate("Form", "确认"))
        self.pushButton_2.setText(_translate("Form", "退出"))
        self.label.setText(_translate("Form", "ISBN"))
        self.label_2.setText(_translate("Form", "作者"))

    def searchbook(self):
        isbn = self.lineEdit.text()
        author = self.lineEdit_2.text()
        self.model = QtSql.QSqlTableModel()
        self.tableView.setModel(self.model)
        self.db = connect(host='localhost', port=5432, charset='utf8', database='Bookshop_Manage_system', password='root',
                          user='postgres')
        # 创建游标对象
        self.cursor = self.db.cursor()
        sql = "use Bookshop_Manage_system"
        self.cursor.execute(sql)
        while True:
            if len(isbn) is 0 and len(author) is not 0:
                sql = "select book.ISBN,BookName,Author,Price,TotalNum " \
                      "from book left join collectionofbook c on book.ISBN = c.ISBN where author = '%s' " % author
                break
            elif len(isbn) is not 0 and len(author) is 0:
                sql = "select book.ISBN,BookName,Author,Price,TotalNum " \
                      " from book left join collectionofbook c on book.ISBN = c.ISBN where book.ISBN = '%s' " % isbn
                break
            elif len(isbn) is not 0 and len(author) is not 0:
                sql = "select book.ISBN,BookName,Author,Price,TotalNum " \
                " from book left join collectionofbook c on book.ISBN = c.ISBN " \
                      "where book.ISBN = '%s' and author = '%s' " % (isbn, author)
                self.statusbar.showMessage("请输入信息", 2000)

        # 定义SQL语句
        print(isbn, author)
        self.execute_sql(sql)
        data = self.cursor.fetchall()
        row = len(data)
        print(row)
        if row is not 0:
            model = QtGui.QStandardItemModel(row, len(data[0]))
            col = len(data[0])
            for i in range(row):
                for j in range(len(data[0])):
                    if j is not 3 and j is not 4:
                        model.setItem(i, j, QtGui.QStandardItem(data[i][j]))
                    else:
                        if data[i][j] is None:
                            model.setItem(i, j, QtGui.QStandardItem(str(0)))
                        else:
                            model.setItem(i, j, QtGui.QStandardItem(str(data[i][j])))
            self.cursor.close()
            model.setHorizontalHeaderLabels(['ISBN', "书名", "作者", "定价", "存货"])
            self.tableView.setModel(model)
        self.statusbar.showMessage("查询成功！总共查询到" + str(row) + "条数据", 2000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    test_ui = Ui_searchbook()
    test_ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())