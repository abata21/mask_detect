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
import os
import sys
import time

import cv2
import flyr
import socket
import numpy as np

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr,
                           non_max_suppression, scale_coords, strip_optimizer)
from utils.augmentations import letterbox
from utils.torch_utils import select_device


def draw_bbox(img, x1, y1, x2, y2, label, temp):
    color = [(0, 0, 222), (0, 222, 0), (0, 222, 222)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color[label], thickness=2, lineType=cv2.LINE_AA)
    text = f'{mask_classes[label]} {round(float(temp), 1)}\'C'
    text_size = max(round(sum(img.shape) / 2 * 0.003), 2) / 2
    pos = (x1, y1 - 5)
    cv2.putText(img, text, pos, 1, text_size, (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(img, text, pos, 1, text_size, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return img


def draw_debug_img(ir_path, im_rgb, labels, temperatures, bboxes, work_path, date, wr, hr, threshold):
    r = round
    imgt = cv2.imread(ir_path)
    rgb = cv2.resize(im_rgb, (1056, 1440), interpolation=cv2.INTER_CUBIC)
    thermal = cv2.resize(imgt, (1056, 1440), interpolation=cv2.INTER_CUBIC)
    if labels and temperatures and bboxes:
        for label, temp, box in zip(labels, temperatures, bboxes):
            if threshold[0] <= temp <= threshold[1]:
                x1, y1, x2, y2 = r((box[0] * wr) * 2.2), r((box[1] * hr) * 2.25), \
                                 r((box[2] * wr) * 2.2), r((box[3] * hr) * 2.25)
                rgb = draw_bbox(rgb, x1, y1, x2, y2, label, round(temp, 3))
                thermal = draw_bbox(thermal, x1 - 8, y1 + 46, x2 - 8, y2 + 46, label, round(temp, 3))
    combined_img = np.hstack((rgb, thermal))
    combined_img = letterbox(combined_img, new_shape=(1440, 2560), color=(0, 0, 0), auto=False)[0]
    os.makedirs(work_path, exist_ok=True)
    cv2.imwrite(f"{work_path}/{date}.jpg", combined_img)
    cv2.imshow('Thermal', combined_img)
    cv2.waitKey(1)
    LOGGER.info(f"Debug image saved to {colorstr('bold', f'{work_path}/{date}.jpg')}")


@torch.no_grad()
def run(
        weights='weights/epoch249_best.pt',  # model.pt path(s)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=True,  # save results to *.txt
        save_img=True,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        update=False,  # update all models
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup


    def job(job_path):

        # Directories
        in_rgb_path = f'files/show/rgb/{job_path}.jpg'
        in_ir_path = f'files/show/ir/{job_path}.jpg'
        debug_path = f'files/show/debug'

        executor2 = ThreadPoolExecutor()  # Concurrent
        future2 = executor2.submit(flyr.unpack, in_ir_path, )

        # Dataloader
        img0 = cv2.imread(in_rgb_path)
        img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        confidences, bboxes, labels, temperatures = [], [], [], []

        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

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

        imt = future2.result()
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = in_rgb_path, img0.copy()
            imd = im0.copy()  # for make debug image
            rgbh, rgbw = im0.shape[:2]
            wr, hr = (irw / rgbw), (irh / rgbh)
            LOGGER.info(f"{'Class'.ljust(15)} {'Temperature'.ljust(20)} {'Confidence'.ljust(20)} {'BBOX'.ljust(30)}")
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    w, h = xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                    w_ratio, h_ratio = w/h, h/w

                    if w_ratio >= 0.4 and h_ratio >= 0.4:
                        bbox = list(int(v.item()) for v in xyxy)
                        confidences.append(conf)
                        bboxes.append(bbox)
                        labels.append(int(cls.item()))

                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        xt1, yt1, xt2, yt2 = int(x1 * wr) - 8, int(y1 * hr) + 46, int(x2 * wr) - 8, int(y2 * hr) + 46
                        xt1 = 0 if xt1 < 0 else xt1
                        yt2 = irh if yt2 > irh else yt2

                        temperature = imt.celsius[yt1:yt2, xt1:xt2].max()
                        temp_devi = 36.5 - temperature
                        temperature = temperature + (temp_devi * 0.7)
                        temperatures.append(temperature)

                        LOGGER.info(f"{colorstr('bold', mask_classes[int(cls.item())].ljust(15))}"
                                    f"{colorstr('bold', str(round(temperature, 3)).ljust(20))}"
                                    f"{colorstr('bold', str(round(conf.item(), 3)).ljust(20))}"
                                    f"{colorstr('bold', str(bbox).ljust(30))}")

            if debug_img:
                draw_debug_img(in_ir_path, imd, labels, temperatures, bboxes,
                               debug_path, job_path, wr, hr, temp_threshold)

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                LOGGER.info(f"Results saved to {colorstr('bold', debug_path)}")
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    IP = '192.168.50.162'
    PORT = 28282
    SIZE = 1024
    ADDR = (IP, PORT)

    window_name = 'Thermal'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, -1, -1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(ADDR)
        server_socket.listen()
        print('booted')
        while True:
            client_socket, client_addr = server_socket.accept()
            msg = client_socket.recv(SIZE)
            client_socket.close()
            path = str(msg.decode())
            LOGGER.info(f"Input path is {colorstr('bold', path)}")
            job(path)


if __name__ == "__main__":
    mask_classes = ['Without_Mask', 'With_Mask', 'Wrong_Mask']
    debug_img = True

    fever = 35
    temp_threshold = (1, 99)
    irw, irh = 480, 640

    check_requirements(exclude=('tensorboard', 'thop'))
    run(weights='weights/fold_v4/fold1.pt')
