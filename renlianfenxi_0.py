# encoding:utf-8
from aip import AipFace
import cv2
import base64
import os

from pygame import mixer
import time




""" 你的APPID，API_KEY和SECRET_KEY """
APP_ID = '24231305'
API_KEY = '7tvMWQoSE6xPMnz84soifzvj'
SECRET_KEY ='XnlCOx2N1FTFYVOXO4aeQVyQBUE9c2Ys'


cap = cv2.VideoCapture(0)
flag = cap.isOpened()

def music1():
    # pygame.init()
    mixer.init()

    sound = mixer.Sound("D:/tuxiangchuli/music/5951.mp3")
    sound.play()

    time.sleep(2)


def music0():
    # pygame.init()
    mixer.init()

    sound = mixer.Sound("D:/tuxiangchuli/music/12762.mp3")
    sound.play()

    time.sleep(2)


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_client(APP_ID, API_KEY, SECRET_KEY):
    """
    返回client对象
    :param APP_ID:
    :param API_KEY:
    :param SECRET_KEY:
    :return:
    """
    return AipFace(APP_ID, API_KEY, SECRET_KEY)



client = get_client(APP_ID, API_KEY, SECRET_KEY)


def face_analysis():
    global a
    for filename in os.listdir(r'D:/tuxiangchuli/yuantu'):
        #print(filename)
        result = client.match([
            {
                'image': str(base64.b64encode(open('D:/tuxiangchuli/'+str(index)+'.png', 'rb').read()), 'utf-8'),
                'image_type': 'BASE64',
            },
            {
                'image': str(base64.b64encode(open("D:/tuxiangchuli/yuantu/" +str(filename), 'rb').read()), 'utf-8'),
                'image_type': 'BASE64',
            }
        ])

        if result['error_msg'] == 'SUCCESS':
            score = result['result']['score']
            if score >= 90:
                    print('匹配程度'+str(score))
                    a=1
                    print('欢迎光临' + str(filename[:-4]))
                    music1()

            else:
                    a = 0
            break
        else:

            print('错误信息：', result['error_msg'])

    return a



index = 1
a=0





while (index):
    ret, frame = cap.read()
    cv2.imshow("Face Analysis", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv2.imwrite("D:/tuxiangchuli/" + str(index) + ".png", frame)
        # print(cap.get(3))
        # print(cap.get(4))
        # print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")

        a=face_analysis()
        #print(a)
        if a == 0:
            # print(score)
            print('陌生人')
            music0()

        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cap.release()
cv2.destroyAllWindows()




























