# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
import json
import os
import sys
import csv
import time

import flyr
import socket
import requests
import numpy as np

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from requests_toolbelt import MultipartEncoder

import torch

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr, cv2,
                           non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh)
from utils.augmentations import letterbox
from utils.plots import Annotator, colorss, save_one_box
from utils.torch_utils import select_device
import math


def draw_bbox(img, x1, y1, x2, y2, label, temp, conf, class_name):
    color = [(0, 0, 222), (0, 222, 0), (0, 222, 222)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color[label], thickness=2, lineType=cv2.LINE_AA)
    text = f'{round(float(temp), 1)}\'C {conf * 100}%' + ' '+ class_name
    text_size = max(round(sum(img.shape) / 2 * 0.003), 2) / 2
    pos = (x1, y1 - 5)
    cv2.putText(img, text, pos, 1, text_size, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, pos, 1, text_size, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return img



def draw_debug_img(ir_path, im_rgb, labels, temperatures, confidences, bboxes, date_time, work_path, wr, hr, threshold,class_name):
    r = round
    imgt = cv2.imread(ir_path)
    cv2.waitKey(1)
    rgb = cv2.resize(im_rgb, (480, 640))
    thermal = cv2.resize(imgt, (480, 640))
    # rgb = cv2.resize(im_rgb, (1056, 1440))
    # thermal = cv2.resize(imgt, (1056, 1440))
    if labels and temperatures and confidences and bboxes and class_name:
        for label, temp, confidence, box, clss in zip(labels, temperatures, confidences, bboxes, class_name):
            if threshold[0] <= temp <= threshold[1]:
                x1, y1, x2, y2 = r(box[0] * wr), r(box[1] * hr), r(box[2] * wr), r(box[3] * hr)
                rgb = draw_bbox(rgb, x1, y1, x2, y2, label, r(temp, 3), r(confidence.item(), 2), clss)
                thermal = draw_bbox(thermal, x1 + 5, y1-18, x2 + 5, y2-18, label,
                                    r(temp, 3), r(confidence.item(), 2), clss)
    combined_img = np.hstack((rgb, thermal))
    combined_img = letterbox(combined_img, new_shape=(960, 1280), color=(0, 0, 0), auto=False)[0]
    # combined_img = letterbox(combined_img, new_shape=(1440, 2560), color=(0, 0, 0), auto=False)[0]
    os.makedirs(f"{work_path[:-14]}/debug", exist_ok=True)
    cv2.imwrite(f"{work_path[:-14]}/debug/{date_time}.jpg", combined_img)
    LOGGER.info(f"Debug image saved to {colorstr('bold', f'{work_path[:-14]}/debug/{date_time}.jpg')}")
    return combined_img


@torch.no_grad()
def run(
        weights='weights/epoch249_best.pt',  # model.pt path(s)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_img=True,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup


    def job(job_path, devicecd, date):

        if timer:
            td = time.perf_counter()  #
            ta = time.perf_counter()  #


        # Directories
        in_rgb_path = f'{job_path}/input/photo.jpg'
        in_ir_path = f'{job_path}/input/msx.jpg'
        out_path = f'{job_path}/output'
        debug_path = f'{job_path}/debug'
        save_dir = out_path

        if timer:
            tdl = time.perf_counter()  #

        os.makedirs(f'{debug_path}', exist_ok=True)
        os.makedirs(f'{save_dir}', exist_ok=True)

        executor2 = ThreadPoolExecutor()  # Concurrent
        future2 = executor2.submit(flyr.unpack, in_ir_path, )


        if timer:
            print(time.perf_counter() - tdl)  #
            tdl = time.perf_counter()  #

        # Dataloader
        img0 = cv2.imread(in_rgb_path)
        img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        face_cnt = 0
        face_ids = []
        confidences = []
        bboxes = []
        labels = []
        temperatures = []
        class_name = []

        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

        if timer:
            print(time.perf_counter() - tdl)  #
            print(f'before: {time.perf_counter() - td}')  #
            td = time.perf_counter()  #

        t1 = time.perf_counter()
        im = torch.from_numpy(im)
        im = im.to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time.perf_counter()
        dt[0] += t2 - t1

        # Inference
        pred = model(im)
        t3 = time.perf_counter()
        dt[1] += t3 - t2

        # NMS
        torch.cuda.synchronize()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time.perf_counter() - t3


        if timer:
            print(f'detect: {time.perf_counter() - td}')  #
            td = time.perf_counter()  #

        imt = future2.result()
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = in_rgb_path, img0.copy()
            p = Path(p)  # to Path
            save_path = f'{save_dir}/{p.name}'  # im.jpg
            txt_path = f"{save_dir}/{p.stem}"  # im.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            imd = im0.copy() if debug_img else im0  # for make debug image
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            LOGGER.info(f"{'Class'.ljust(15)} {'Temperature'.ljust(20)} {'Confidence'.ljust(20)} {'BBOX'.ljust(30)}")
            rgbh, rgbw = im0.shape[:2]
            wr, hr = (irw / rgbw), (irh / rgbh)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    w, h = xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                    w_ratio, h_ratio = w/h, h/w

                    if w_ratio >= 0.4 and h_ratio >= 0.4:
                        bbox = list(int(v.item()) for v in xyxy)
                        # print(bbox)
                        face_ids.append(f'face_{str(face_cnt).zfill(3)}')
                        confidences.append(conf)
                        bboxes.append(bbox)
                        labels.append(int(cls.item()))

                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        xt1, yt1, xt2, yt2 = int(x1 * wr) - 3, int(y1 * hr) + 18, int(x2 * wr) - 3, int(y2 * hr) + 18
                        xt1 = 0 if xt1 < 0 else xt1
                        yt2 = irh if yt2 > irh else yt2

                        # Distance
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        width, height = xywh[2:]
                        diag = (((width ** 2) + (height ** 2)) ** (1 / 2))
                        dist = ((-(math.log10(diag))) * 5) ** 3
                        distance = dist * (33.99 / dist + 2.0585) if dist <= 70.624 else dist * (64 / dist + 1.633574)
#
#                         # Temperature
                        yt2 = 640 if yt2 > 640 else yt2
                        temp = imt.celsius[yt1:yt2, xt1:xt2].max()
                        temperature = 330 / (temp + 30) + 31.37144 if distance <= 272.182 else 10.5 / (
                                    temp - 265.6) + 36.40
                        temperatures.append(temperature)
                        class_name.append(mask_classes[int(cls.item())])
                        LOGGER.info(f"{colorstr('bold', mask_classes[int(cls.item())].ljust(15))} "
                                    f"{colorstr('bold', str(round(temperature, 2)).ljust(20))} "
                                    f"{colorstr('bold', str(round(conf.item(), 2)).ljust(20))} "
                                    f"{colorstr('bold', str(bbox).ljust(30))}")

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{conf:.2f}')
                            annotator.box_label(xyxy, label, color=colorss[c])
                            if save_crop:
                                save_one_box(xyxy, imc, file=f"{save_dir}/crops/{names[c]}/{p.stem}.jpg", BGR=True)
                        face_cnt += 1

            events = []

            #  기존 코드
            with open(f'{out_path}/output.csv', 'w', newline='') as csv_out:
                log = csv.writer(csv_out)
                header = ['', 'face_id', 'class', 'temperature', 'confidence', 'bbox', 'event']
                log.writerow(header)
                index = 0

                for face, label, temp, confidence, box in zip(face_ids, labels, temperatures, confidences, bboxes):
                    row = [index, face, mask_classes[label], round(temp, 3), round(confidence.item(), 3), box]


                    if label == 0 or label == 2 or temp > fever:
                        event_list = []
                        if label == 0 or label == 2:
                            event_list.append(mask_classes[label])
                        if temp > fever:
                            event_list.append('fever')
                        event = {'desc': ' '.join(str(s) for s in event_list),
                                 'bbox': ' '.join(str(s) for s in box),
                                 'temp': round(temp, 3)}

                        events.append(event)
                        row.append(event)

                    log.writerow(row)
                    index += 1



            if send_event and events:
                # resz = cv2.imread(in_rgb_path)
                # resz2 = cv2.resize(resz, (480,640))
                # cv2.imwrite(f'{job_path}/input/photo2.jpg',resz2)
                # resize_path = f'{job_path}/input/photo2.jpg'
                files = (f'{devicecd}_{date}.jpg', open(in_rgb_path, 'rb'), 'image/jpeg')
                # files = (f'{devicecd}_{date}.jpg', open(resize_path, 'rb'), 'image/jpeg')
                data = MultipartEncoder(fields={'deviceCd': devicecd,
                                                'file': files,
                                                'domainCd': 'BTP',
                                                'data': json.dumps([v for v in events]),
                                                'date': f'20{date[:2]}-{date[2:4]}-{date[4:6]} '
                                                        f'{date[7:9]}:{date[9:11]}:{date[11:13]}',
                                                'deviceGrpCd': 'feverCam'})
                r = requests.post(thub_address, data=data, headers={'Content-Type': data.content_type})

            if timer:
                print(f'after: {time.perf_counter() - td}')  #
                print(f'all process: {time.perf_counter() - ta}')  #

            if send_event and events:
                print(r.text)

            im0 = annotator.result()
            if save_img:
                cv2.imwrite(save_path, im0)


            if debug_img:
                # print(class_name)
                combine_img = draw_debug_img(in_ir_path, imd, labels, temperatures, confidences, bboxes, now, job_path, wr, hr, temp_threshold,class_name)



            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
                print('one_cycle_end : @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        return combine_img

    IP = '192.168.50.136'
    PORT = 28283
    SIZE = 1024
    ADDR = (IP, PORT)


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # start_time = time.time()
        server_socket.bind(ADDR)
        server_socket.listen()
        print('booted')
        cv2.namedWindow('test')
        # print('end_time', time.time()-start_time)

        while True:
            start_time = time.time()
            client_socket, client_addr = server_socket.accept()
            msg = client_socket.recv(SIZE)
            # print(msg)
            client_socket.close()
            path = str(msg.decode())
            # print("path", path)
            LOGGER.info(f"Input path is {colorstr('bold', path)}")
            deviceid = path.split('/')[-2]
            now = path.split('/')[-1]
            combine_img = job(path, deviceid, now)
            cv2.imshow('test', combine_img)
            cv2.waitKey(1)
            print('end_time', time.time() - start_time)

if __name__ == "__main__":

    mask_classes = ['Without_Mask', 'With_Mask', 'Wrong_Mask']
    # thub_address = 'http://192.168.50.136:7878/' #'http://192.168.50.136:7878/', 'http://ics.thubiot.com/api/ics/rcvFeverCamData'
    thub_address = 'http://ics.thubiot.com/api/ics/rcvFeverCamData'

    fever = 35
    temp_threshold = (0, 42)

    send_event = True #False
    debug_img = True
    timer = False
    save_log = True

    irw, irh = 480, 640

    check_requirements(exclude=('tensorboard', 'thop'))
    run(weights='weights/fold_v4/fold1.pt')
