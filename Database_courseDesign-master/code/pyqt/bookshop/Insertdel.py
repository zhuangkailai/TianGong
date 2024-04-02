# -*- coding: utf-8 -*-


from PyQt5 import QtCore, QtGui, QtWidgets
from pymysql import *

class Ui_delbook(object):
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

    def setupUi(self, delbook):
        delbook.setObjectName("delbook")
        delbook.resize(515, 336)
        self.pushButton_2 = QtWidgets.QPushButton(delbook)
        self.pushButton_2.setGeometry(QtCore.QRect(110, 230, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(delbook)
        self.pushButton_3.setGeometry(QtCore.QRect(320, 230, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.widget = QtWidgets.QWidget(delbook)
        self.widget.setGeometry(QtCore.QRect(80, 140, 352, 30))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)

        self.retranslateUi(delbook)
        self.pushButton.clicked.connect(self.lineEdit.clear)
        self.pushButton_3.clicked.connect(delbook.close)
        QtCore.QMetaObject.connectSlotsByName(delbook)
        self.pushButton_2.clicked.connect(lambda: self.insert_del())

    def retranslateUi(self, delbook):
        _translate = QtCore.QCoreApplication.translate
        delbook.setWindowTitle(_translate("delbook", "删除书籍"))
        self.pushButton_2.setText(_translate("delbook", "确认"))
        self.pushButton_3.setText(_translate("delbook", "退出"))
        self.label.setText(_translate("delbook", "ISBN"))
        self.pushButton.setText(_translate("delbook", "清空"))

    def insert_del(self):
        isbn = self.lineEdit.text()
        self.db = connect(host='localhost', port=5432, charset='utf8', database='Bookshop_Manage_system', password='root',
                          user='postgres')
        # 创建游标对象
        self.cursor = self.db.cursor()
        sql = "use Bookshop_Manage_system"
        self.cursor.execute(sql)
        sql = "delete from book where ISBN = '%s' " % (isbn)
        self.execute_sql(sql)
        self.cursor.connection.commit()
        self.db.close()

