# -*- coding: utf-8 -*-

import sys
from pymysql import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication


class Ui_insert_employee(object):
    def execute_sql(self, sql):
        try:
            # 执行数据库操作
            self.cursor.execute(sql)
            # 事务提交
            self.db.commit()
        except Exception as err:
            # 事务回滚
            print("hello err")
            self.db.rollback()
            print("SQL执行错误，原因：", err)

    def setupUi(self, insert_employee):
        insert_employee.setObjectName("insert_employee")
        insert_employee.resize(781, 523)
        self.pushButton = QtWidgets.QPushButton(insert_employee)
        self.pushButton.setGeometry(QtCore.QRect(540, 80, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(insert_employee)
        self.pushButton_2.setGeometry(QtCore.QRect(540, 150, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(insert_employee)
        self.pushButton_3.setGeometry(QtCore.QRect(540, 220, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.widget = QtWidgets.QWidget(insert_employee)
        self.widget.setGeometry(QtCore.QRect(120, 70, 331, 221))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.radioButton = QtWidgets.QRadioButton(self.widget)
        self.radioButton.setObjectName("radioButton")
        self.gridLayout.addWidget(self.radioButton, 2, 1, 1, 1)
        self.radioButton_2 = QtWidgets.QRadioButton(self.widget)
        self.radioButton_2.setObjectName("radioButton_2")
        self.gridLayout.addWidget(self.radioButton_2, 2, 2, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 2)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 2)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 1, 1, 2)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout.addWidget(self.lineEdit_5, 4, 1, 1, 2)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.lineEdit_6, 5, 1, 1, 2)

        self.statusbar = QtWidgets.QStatusBar(insert_employee)
        self.statusbar.setObjectName("statusbar")
        insert_employee.setStatusBar(self.statusbar)

        self.sex = None
        self.radioButton.clicked.connect(lambda: self.chooseSex_male())
        self.radioButton_2.clicked.connect(lambda: self.chooseSex_female())
        self.retranslateUi(insert_employee)
        self.pushButton_2.clicked.connect(self.lineEdit.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_2.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_4.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_5.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_6.clear)
        self.pushButton_3.clicked.connect(insert_employee.close)
        QtCore.QMetaObject.connectSlotsByName(insert_employee)

        self.pushButton.clicked.connect(lambda: self.insertEmployee())

    def retranslateUi(self, insert_employee):
        _translate = QtCore.QCoreApplication.translate
        insert_employee.setWindowTitle(_translate("insert_employee", "Form"))
        self.pushButton.setText(_translate("insert_employee", "确认"))
        self.pushButton_2.setText(_translate("insert_employee", "清空"))
        self.pushButton_3.setText(_translate("insert_employee", "退出"))
        self.label_4.setText(_translate("insert_employee", "年龄"))
        self.label_6.setText(_translate("insert_employee", "工资"))
        self.label.setText(_translate("insert_employee", "雇员ID"))
        self.label_3.setText(_translate("insert_employee", "性别"))
        self.label_5.setText(_translate("insert_employee", "电话"))
        self.label_2.setText(_translate("insert_employee", "姓名"))
        self.radioButton.setText(_translate("insert_employee", "男"))
        self.radioButton_2.setText(_translate("insert_employee", "女"))


    def chooseSex_male(self):
        self.sex = "男"

    def chooseSex_female(self):
        self.sex = "女"

    def insertEmployee(self):
        id = self.lineEdit.text()
        name = self.lineEdit_2.text()
        if self.sex is None:
            self.statusbar.showMessage("请选择性别", 2000)
        else:
            salary = self.lineEdit_6.text()
            tel = self.lineEdit_5.text()
            age = self.lineEdit_4.text()
            print(id, name, self.sex, salary, tel, age)
            self.db = connect(host='localhost', port=5432, charset='utf8', database='Bookshop_Manage_system', password='root',
                          user='postgres')
            # 创建游标对象
            self.cursor = self.db.cursor()

            sql = "use Bookshop_Manage_system"
            self.cursor.execute(sql)
            sql = "select * from employee"
            self.execute_sql(sql)
            data = self.cursor.fetchall()
            original_row = len(data)
            sql = "insert into employee VALUES (" \
                  " '%d','%s','%s','%d','%s','%d') " %(int(id), name, self.sex, int(age), tel, int(salary))
            self.execute_sql(sql)
            sql = "select * from employee"
            self.execute_sql(sql)
            data = self.cursor.fetchall()
            changed_row = len(data)
            if changed_row == original_row+1:
                self.statusbar.showMessage("增加成功", 2000)
            else:
                self.statusbar.showMessage("增加失败",2000)



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
    test_ui = Ui_insert_employee()
    test_ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
