# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import psycopg2
import time  # 引入time模块
import argparse
import os
import sys
from pathlib import Path
import linecache
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import socket
import serial
from Get_image_demo3 import gesture
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#设施舵机参数
portx="COM8"
timex=5
ser=serial.Serial(portx,115200,timeout=timex)

#树莓派向电脑传ID
def server():
    HOST_IP = "192.168.1.102"
    # HOST_IP =socket.gethostname() # 获取本地主机名
    HOST_PORT = 8864
    print("Starting socket: TCP...")
    # 1.create socket object:socket=socket.socket(family,type)
    socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("TCP server listen @ %s:%d!" % (HOST_IP, HOST_PORT))
    host_addr = (HOST_IP, HOST_PORT)
    # 2.bind socket to addr:socket.bind(address)
    socket_tcp.bind(host_addr)
    # 3.listen connection request:socket.listen(backlog)
    socket_tcp.listen(3)
    # 4.waite for client:connection,address=socket.accept()
    socket_con, (client_ip, client_port) = socket_tcp.accept()
    print("Connection accepted from %s." % client_ip)

    socket_con.send(str.encode("Welcome to System! Please input ypur ID:"))
    # 5.handle
    '''
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11,GPIO.OUT)
    '''
    print("Receiving package of ID......")
    return socket_con

def take_photo(photoname):
    cap = cv2.VideoCapture(0)
    flag = cap.isOpened()

    index = 1
    while (flag):
        ret, frame = cap.read()
        cv2.imshow("Capture_Paizhao", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):  # 按下s键，进入下面的保存图片操作
            cv2.imwrite("data/takephoto/" + photoname +'.jpg', frame)
            print(cap.get(3))
            print(cap.get(4))
            print("save" + str(index) + ".jpg successfuly!")
            print("-------------------------")
            index += 1
        elif k == ord('q'):  # 按下q键，程序退出
            break
    cap.release()
    cv2.destroyAllWindows()

    photo_name = "data/takephoto/" + photoname + '.jpg'
    return photo_name

def inter(ID,judge_hat,judge_mask):

    Date = str(time.strftime("%Y-%m-%d", time.localtime()))

    Time = str(time.strftime("%H:%M:%S", time.localtime()))

    print(Date)
    print(Time)

    conn = psycopg2.connect(database="postgres", user="postgres", password="root", host="127.0.0.1", port="5432")
    print("Opened database successfully")

    cur = conn.cursor()

    # cur.execute("INSERT INTO results (number,date,judge,time) \
    # VALUES ('20210104','2021-12-19','True',NOW())");

    cur.execute("INSERT INTO inter VALUES (%s, %s, %s,%s,%s)", (ID, Date, Time,judge_hat,judge_mask))

    # cur.execute("INSERT INTO vipcustomer (customerno,viplevel) \
    # VALUES (12001, 1)");

    conn.commit()
    print("Records created successfully")

    conn.close()

def write_object(number,judge):
    Date = str(time.strftime("%Y-%m-%d", time.localtime()))

    Time = str(time.strftime("%H:%M:%S", time.localtime()))

    print(Date)
    print(Time)

    conn = psycopg2.connect(database="postgres", user="postgres", password="root", host="127.0.0.1", port="5432")
    print("Opened database successfully")

    cur = conn.cursor()

    # cur.execute("INSERT INTO results (number,date,judge,time) \
    # VALUES ('20210104','2021-12-19','True',NOW())");
    print(number)
    print(judge)

    cur.execute("INSERT INTO results VALUES (%s,%s,%s,%s)", (number,Date,Time,judge))

    # cur.execute("INSERT INTO vipcustomer (customerno,viplevel) \
    # VALUES (12001, 1)");

    conn.commit()
    print("Records created successfully");
    conn.close()

def judge ():
    linecache.checkcache(filename=None)
    text1 = linecache.getline(r'result.txt', 1)
    print(text1)
    text2 = linecache.getline(r'result.txt', 2)
    print(text2)
    Hat = 'helmet'
    Mask = 'Yes'
    if Hat in text1:
        print("戴头盔")
        judge_hat = "1"
    else:
        print("没戴头盔")
        judge_hat = "0"

    if Mask in text2:
        print("戴口罩")
        judge_mask = "1"
    else:
        print("没戴口罩")
        judge_mask = "0"

    with open("result.txt", 'r+') as file:#清空文件夹
        file.truncate(0)
    linecache.checkcache(filename=None)
    return judge_hat,judge_mask

def judge_object():

    linecache.checkcache(filename=None)
    text1 = linecache.getline(r'result.txt', 1)
    print(text1)

    Sterling = 'Sterling'
    apple = "apple"
    stop = "cat"

    if Sterling in text1:
        print("正品")
        judge_object = "True"

    elif Sterling not in text1:
         print("次品")
         judge_object = "False"

    with open("result.txt", 'r+') as file:  # 清空文件夹
        file.truncate(0)

    linecache.checkcache(filename=None)

    return judge_object

def duoji(judge_object):

    if judge_object=="True":
        # 120
        myinput = bytes([0xFA, 0xAF, 0x04, 0x01, 0xF0, 0x64, 0x00, 0x01, 0x5A, 0xED])
        ser.write(myinput)  # 写数据
        time.sleep(3)

        # 0
        myinput = bytes([0xFA, 0xAF, 0x04, 0x01, 0x78, 0x64, 0x00, 0x01, 0xE2, 0xED])
        ser.write(myinput)
        time.sleep(1)

    elif judge_object == "False":
        # -120
        myinput = bytes([0xFA, 0xAF, 0x04, 0x01, 0x01, 0x64, 0x00, 0x01, 0x6B, 0xED])
        ser.write(myinput)
        time.sleep(3)

        # 0
        myinput = bytes([0xFA, 0xAF, 0x04, 0x01, 0x78, 0x64, 0x00, 0x01, 0xE2, 0xED])
        ser.write(myinput)
        time.sleep(1)

def judge_ID(ID):
    conn = psycopg2.connect(database="postgres", user="postgres", password="root", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    select_sql = "select username from logadmin WHERE username=%s"
    cur.execute(select_sql, [ID])

    result = cur.fetchone()
    print(result)
    if result is None:
        ID_judge=0
    else:
        ID_judge=1

    return ID_judge



@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            # check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')



            #将识别结果保存到指定txt文件
            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')
            with open("result.txt", "a") as f:
                f.write(f'{s}'+'\n')  # 这句话自带文件关闭功能，不需要再写f.close()



            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)



def parse_opt(model_location,photo_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / model_location, help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / photo_name, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

def begin_main():
    #从树莓派接收ID
    socket = server()

    while 1:
        ID = bytes.decode(socket.recv(512))
        ID_judge=judge_ID(ID)
        if ID_judge==1:
            ID_judge = '1'
            socket.send(str.encode(ID_judge))
            break
        elif ID_judge==0:
            print("该员工不存在")
            ID_judge = '0'
            socket.send(str.encode(ID_judge))


    print("ID已经收到，开启摄像头，准备拍照检测口罩头盔")
    photoname = 'face'+ str(time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()))
    photo_name = take_photo(photoname)
    print("照片已拍好，检测是否佩戴齐全")

    #检测是否带安全头盔
    lacation = 'weights/helmet_head_person_l.pt'
    opt = parse_opt(lacation,photo_name)
    main(opt)

    print("stop")

    #检测是否戴口罩
    lacation = 'runs/train/exp_mask/weights/best.pt'
    opt = parse_opt(lacation,photo_name)
    main(opt)

    while 1:
        judge_hat, judge_mask = judge()
        if judge_hat == "1" and judge_mask == "1":
            print("已正确佩戴口罩和安全头盔")
            break
        elif judge_mask =="0" or judge_hat =="0" :
            if judge_mask =="0":
                print("请佩戴口罩")
            if judge_hat == "0":
                print("请佩戴安全头盔")

            photoname = 'face'+str(time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()))
            photo_name = take_photo(photoname)

            # 检测是否带安全头盔
            lacation = 'weights/helmet_head_person_l.pt'
            opt = parse_opt(lacation, photo_name)
            main(opt)

            # 检测是否戴口罩
            lacation = 'runs/train/exp_mask/weights/best.pt'
            opt = parse_opt(lacation, photo_name)
            main(opt)

    inter(ID, judge_hat, judge_mask)  # 将检测结果保存到数据库


    print("人员信息已存入数据库，开始拍照检测手势")
    ready = 'taking_photo_shoushi'
    socket.send(str.encode(ready))



    photoname = 'shoushi'+str(time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()))
    photo_name = take_photo(photoname)
    print(photo_name)
    shoushi_result=gesture(photo_name,photo_name)
    print(shoushi_result)


    print("识别到开启手势，给树莓派发开启电机消息")
    begin_dianji = 'begin_dianji'
    socket.send(str.encode(begin_dianji))


    while 1:
        have_object = bytes.decode(socket.recv(512))
        print(have_object)

        print("收到树莓派检测到物品的消息，开始拍照，准备识别")
        #拍物品照片
        photoname = 'object'+str(time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime()))
        photo_name = take_photo(photoname)
        # 检测正品
        lacation = 'runs/train/exp/weights/best.pt'
        #lacation = 'weights/yolov5x.pt'
        opt = parse_opt(lacation, photo_name)
        main(opt)
        object_judge = judge_object()




        write_object(photoname,object_judge)
        duoji(object_judge)
        print("该物品检测完毕")
        finish = 'finish'
        socket.send(str.encode(finish))





    #print("整条流水线结束")


if __name__ == "__main__":
    begin_main()