# -*- coding: utf-8 -*-
# !/usr/bin/env python
import urllib.parse
import requests
import urllib
import base64
import json
import time
import sys



# client_id 为官网获取的AK， client_secret 为官网获取的SK


# 获取token
def get_token_key():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    client_id = 'mqVW23EsrCw2ehKR6bsufOpB'
    client_secret = 'ycyyusWKyg4y8MCi0CIkjOFsM4A4wlI9'
    url = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials' \
          f'&client_id={client_id}&client_secret={client_secret}'
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    res = requests.post(url, headers=headers)
    token_content = res.json()
    assert "error" not in token_content, f"{token_content['error_description']}"
    token_key = token_content['access_token']
    return token_key


def get_hand_info(image_base64, token_key):
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/gesture"
    params_d = dict()
    params_d['image'] = str(image_base64, encoding='utf-8')
    access_token = token_key
    request_url = request_url + "?access_token=" + access_token
    res = requests.post(url=request_url,
                        data=params_d,
                        headers={'Content-Type': 'application/x-www-form-urlencoded'})
    data = res.json()
    assert 'error_code' not in data, f'Error: {data["error_msg"]}'
    return data


# 画出手势识别结果
def draw_gestures(originfilename, gestures, resultfilename):
    from PIL import Image, ImageDraw

    image_origin = Image.open(originfilename)
    draw = ImageDraw.Draw(image_origin)

    for gesture in gestures:
        draw.rectangle(
            (gesture['left'], gesture['top'], gesture['left'] + gesture['width'], gesture['top'] + gesture['height']),
            outline="red")
        draw.text((gesture['left'], gesture['top']), gesture['classname'], "blue")

    image_origin.save(resultfilename, "JPEG")


# 手势识别
# filename:原图片名（本地存储包括路径）
def gesture(filename, resultfilename):
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/gesture"
    print(filename)
    # 二进制方式打开图片文件
    f = open(filename, 'rb')
    img = base64.b64encode(f.read())

    params = dict()
    params['image'] = img
    params = urllib.parse.urlencode(params).encode("utf-8")
    # params = json.dumps(params).encode('utf-8')

    access_token = get_token_key()
    begin = time.perf_counter()
    request_url = request_url + "?access_token=" + access_token
    request = urllib.request.Request(url=request_url, data=params)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request)
    content = response.read()
    end = time.perf_counter()

    print('处理时长:' + '%.2f' % (end - begin) + '秒')
    if content:
        # print(content)
        content = content.decode('utf-8')
        # print(content)
        data = json.loads(content)
        data_str = str(json.loads(content))
        print(data)
        print(data_str)
        result = data['result']

        classname = data.get('result')
        next = classname[0]
        last = next.get('classname')
        #last1 = next.get('width')

        #print(last)
        #print(last1)

        #print(result)
        with open("class.txt", "w+") as f:
            f.write(f'{result}')  # 这句话自带文件关闭功能，不需要再写f.close()

        draw_gestures(filename, result, resultfilename)

        return last


if __name__ == "__main__":

    a =gesture('shoushi.jpg', 'shoushi_result.jpg')
    print(a)
