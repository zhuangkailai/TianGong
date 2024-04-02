
from tkinter import *
import psycopg2
from detect import begin_main


def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("注册")
    register_screen.geometry("300x250")
    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
    Label(register_screen, text="请输入", bg="white").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="用户名 * ")
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_lable = Label(register_screen, text="密码 * ")
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="注册", width=10, height=1, bg="white", command=register_user).pack()

def login():

    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("登录")
    login_screen.geometry("300x250")
    Label(login_screen, text="请输入").pack()
    Label(login_screen, text="").pack()
    global username_verify
    global password_verify
    username_verify = StringVar()
    password_verify = StringVar()
    global username_login_entry
    global password_login_entry
    Label(login_screen, text="用户名 * ").pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="密码 * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show='*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="登录", width=10, height=1, command=login_verify).pack()

def register_user():
    username_info = username.get()
    password_info = password.get()
    conn = psycopg2.connect(database="postgres", user="postgres", password="root", host="127.0.0.1", port="5432")
    print("Opened database successfully")

    cur = conn.cursor()

    select_sql = "select username from logadmin WHERE username=%s"
    cur.execute(select_sql, [username_info])
    result = cur.fetchone()
    if result is None:
        try:
            cur.execute("INSERT INTO logadmin VALUES (%s,%s)", (username_info, password_info))
            print("插入成功")
            conn.commit()
            cur.close()
            conn.close()
        except:
            conn.rollback()
        username_entry.delete(0, END)
        password_entry.delete(0, END)
        Label(register_screen, text="注册成功！", fg="green", font=("宋体", 11)).pack()
    else:
        is_been_registered()

def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)
    conn = psycopg2.connect(database="postgres", user="postgres", password="root", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    select_sql = "select password from logadmin WHERE username=%s"
    cur.execute(select_sql,[username1])

    result = cur.fetchone()
    print(result)
    if result is None:
        user_not_found()
    else :
        if result[0] == password1:
            login_sucess()
        elif result[0] != password1:
            password_not_recognised()

def login_sucess():
    global login_success_screen
    login_success_screen = Toplevel(login_screen)
    login_success_screen.title("恭喜")
    login_success_screen.geometry("150x100")
    Label(login_success_screen, text="登录成功！").pack()
    Button(login_success_screen, text="确定", command=delete_login_success).pack()

def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("失败")
    password_not_recog_screen.geometry("150x100")
    Label(password_not_recog_screen, text="密码错误 ").pack()
    Button(password_not_recog_screen, text="确定", command=delete_password_not_recognised).pack()

def is_been_registered():
    global is_been_registered_screen
    is_been_registered_screen = Toplevel(register_screen)
    is_been_registered_screen.title("失败")
    is_been_registered_screen.geometry("150x100")
    Label(is_been_registered_screen, text="该用户名已注册").pack()
    Button(is_been_registered_screen, text="确定", command=delete_is_been_registered_screen).pack()

def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("失败")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="该用户未注册").pack()
    Button(user_not_found_screen, text="确认", command=delete_user_not_found_screen).pack()

def delete_login_success():
    login_success_screen.destroy()
    login_screen.destroy()
    main_screen.destroy()
    begin_main()

def delete_password_not_recognised():
    password_not_recog_screen.destroy()

def delete_user_not_found_screen():
    user_not_found_screen.destroy()

def delete_is_been_registered_screen():
    is_been_registered_screen.destroy()

def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("300x250")
    main_screen.title("登陆界面")
    Label(text="欢迎您", bg="white", width="300", height="2", font=("宋体", 13)).pack()
    Label(text="").pack()
    Button(text="登录系统", height="2", width="30", command=login).pack()
    Label(text="").pack()
    Button(text="添加员工", height="2", width="30", command=register).pack()
    main_screen.mainloop()

if __name__ == "__main__":

    main_account_screen()
