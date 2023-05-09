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

import argparse
import os
import sys
from pathlib import Path

import csv, flyr, socket, datetime, requests

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from threading import Thread
from PIL import Image


@torch.no_grad()
def run(
        weights='weights/epoch249_0718.pt',  # model.pt path(s)
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
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        line_thickness=0.3,  # bounding box thickness (pixels)
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

    def job(in_rgb_path, in_ir_path, out_path, w_r, h_r, device_id, date):
        nonlocal visualize

        # Directories
        save_dir = out_path
        os.makedirs(f'{save_dir}', exist_ok=True)
        source = check_file(str(in_rgb_path))
        imt = flyr.unpack(in_ir_path)
        # Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        face_cnt = 0
        face_ids = []
        confidences = []
        bboxes = []
        labels = []
        temperatures = []

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Get Ratio
            imo = Image

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = f'{save_dir}/{p.name}'  # im.jpg
                txt_path = f"{save_dir}/{p.stem}"  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        print(f"\nBBOX: {' '.join(str(int(v.item())) for v in xyxy)}\nClass: {mask_classes[int(cls.item())]}")
                        bbox = list(int(v.item()) for v in xyxy)
                        face_ids.append(f'face_{str(face_cnt).zfill(3)}')
                        confidences.append(conf)
                        bboxes.append(bbox)
                        labels.append(int(cls.item()))

                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        temperature = imt.celsius[int(y1 * h_r):int(y2 * h_r), int(x1 * w_r):int(x2 * w_r)].max()
                        temperatures.append(temperature)

                        with open(f'{out_path}/output.csv', 'w', newline='') as csv_out:
                            log = csv.writer(csv_out)
                            header = ['', 'face_id', 'class', 'temperature', 'bbox', 'confidence', 'event']
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
                                    event = {'desc': ', '.join(event_list),
                                             'bbox': box,
                                             'temp': round(temp, 3)}
                                    row.append(event)

                                    if send_event:
                                        requests.post(url_to,
                                                      json={'device_id': device_id, 'event': event, 'date': date})
                                log.writerow(row)
                                print(row)
                                index += 1

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            label = None if hide_labels else (names[c] if hide_conf else f'{conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors[c])
                        if save_crop:
                            save_one_box(xyxy, imc, file=f"{save_dir}/crops/{names[c]}/{p.stem}.jpg", BGR=True)

                # Stream results
                im0 = annotator.result()
                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    IP = '192.168.50.135'
    PORT = 28283
    SIZE = 1024
    ADDR = (IP, PORT)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(ADDR)
        server_socket.listen()
        print('booted')
        while True:
            client_socket, client_addr = server_socket.accept()
            msg = client_socket.recv(SIZE)
            # client_socket.sendall("done".encode())
            client_socket.close()
            path = str(msg.decode())
            print(f'\"{path}\"')
            deviceid = path.split('/')[-2]
            now = path.split('/')[-1]
            input_rgb_path = f'{path}/input/photo.jpg'
            input_ir_path = f'{path}/input/msx.jpg'
            output_path = f'{path}/output'
            imrgb = Image.open(input_rgb_path)
            imir = Image.open(input_ir_path)
            w1, h1 = imrgb.size
            w2, h2 = imir.size
            wr, hr = (w2 / w1), (h2 / h1)
            t = Thread(target=job, args=(input_rgb_path, input_ir_path, output_path, wr, hr, deviceid, now))  # Multi Thread
            t.start()

    # while True:
    #     input_rgb_path = 'data/input/photo.jpg'
    #     input_ir_path = 'data/input/msx.jpg'
    #     deviceid = 'sh'
    #     imrgb = Image.open(input_rgb_path)
    #     imir = Image.open(input_ir_path)
    #     w1, h1 = imrgb.size
    #     w2, h2 = imir.size
    #     wr, hr = (w2 / w1), (h2 / h1)
    #     now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S.%f')[:-4]
    #     output_path = f"data/output/{now}"
    #     input('Done')
    #     th = Thread(target=job, args=(input_rgb_path, input_ir_path, output_path, wr, hr, deviceid, now))
    #     th.start()


if __name__ == "__main__":

    mask_classes = ['Without_Mask', 'With_Mask', 'Wrong_Mask']
    url_to = 'http://192.168.50.163:5000/'
    send_event = False
    fever = 30

    check_requirements(exclude=('tensorboard', 'thop'))
    run()
