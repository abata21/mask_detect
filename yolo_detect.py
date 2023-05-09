#yolo
import json
import os
import sys
import csv
import flyr
import socket
import requests
import numpy as np
import math
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
#multiprocess & flask
import torch.multiprocessing
import app as flask_app
import time

class yolo_detect():
    def __init__(self):
        self.yolo_weight = 'C:/Users/abc/Desktop/신종감염병/yolov5/weights/fold_v4/fold1.pt'
        self.img_size = (640,640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = '0'
        self.view_img = False
        self.save_txt = True
        self.save_conf = False
        self.save_crop = False
        self.save_img = True
        self.classes = None
        self.agnostic_nms = True
        self.update = False
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False
        self.half = False
        self.dnn = False

        self.irh = 640
        self.irw = 480
        self.fever = 37.5
        self.temp_threshold = (0, 42)
        self.events = []
        self.send_event = False  # False
        self.debug_img = True
        self.timer = False
        self.save_log = False

        self.mask_classes = ['Without_Mask', 'With_Mask', 'Wrong_Mask']
        # self.thub_address = 'http://192.168.50.136:7878/'
        self.thub_address = 'http://ics.thubiot.com/api/ics/rcvFeverCamData'

        self.mp = torch.multiprocessing.get_context('spawn')
        self.model_init()

    def model_init(self):
        self.detect_model_init()

    def detect_model_init(self):
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.yolo_weight, device=self.device, dnn=self.dnn, fp16=self.half)
        self.stride, self.name, self.pt = self.model.stride, self.model.names, self.model.pt
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.img_size))

    def draw_bbox(self,img, x1, y1, x2, y2, label, temp, conf, class_name):
        color = [(0, 0, 222), (0, 222, 0), (0, 222, 222)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color[label], thickness=2, lineType=cv2.LINE_AA)
        text = f'{round(float(temp), 1)}\'C {conf * 100}%' + ' ' + class_name
        text_size = max(round(sum(img.shape) / 2 * 0.003), 2) / 2
        pos = (x1, y1 - 5)
        cv2.putText(img, text, pos, 1, text_size, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, text, pos, 1, text_size, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        return img

    def cv2_debug_img(self,ir_img, imd, labels, temperatures, confidences, bboxes, now, folder_path, wr, hr, temp_threshold,class_name):
        try:
            cv2_ir_read = cv2.imread(ir_img)
            rgb = cv2.resize(imd, (480,640))
            ir = cv2.resize(cv2_ir_read, (480,640))

            if labels and temperatures and confidences and bboxes and class_name:
                for label, temp, confidence, box, cls in zip (labels, temperatures, confidences, bboxes, class_name):
                    if temp_threshold[0] <= temp <= temp_threshold[1]:
                        x1, y1, x2, y2 = round(box[0] * wr), round(box[1] * hr),round(box[2] * wr), round(box[3] * hr)
                        rgb = self.draw_bbox(rgb, x1, y1, x2, y2, label, round(temp, 3), round(confidence.item(), 2), cls)
                        ir = self.draw_bbox(ir, x1+5, y1-18, x2+5, y2-18,label,round(temp, 3), round(confidence.item(), 2), cls)
            # combined_img = np.hstack(rgb, ir)
            # combined_img = letterbox(combined_img, new_shape=(960, 1280), color=(0, 0, 0), auto=False)[0]
            combined_img = cv2.hconcat([rgb, ir])
            os.makedirs(f"{self.first_path}/debug", exist_ok=True)
            cv2.imwrite(f"{self.first_path}/debug/{now}.jpg", combined_img)
            window_name = 'test'
            cv2.imshow(window_name, combined_img)
            cv2.waitKey(1)
            LOGGER.info(f"Debug image saved to {colorstr('bold', f'{self.first_path}/debug/{now}.jpg')}")
        except Exception:
            print('cv2_debug_img_error')

    def yolo_model_and_thub_send(self):
        IP = '192.168.50.136'
        PORT = 28283
        SIZE = 1024
        ADDR = (IP, PORT)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(ADDR)
            server_socket.listen()
            #탐지 부분 반복 -> 반복이유는 socket으로 받는 img path가 계속해서 달라지기 때문과 계속해서 img가 들어와서 이렇게 하였음
            while True:
                client_socket, client_addr = server_socket.accept()
                msg = client_socket.recv(SIZE)
                msg2 = str(msg.decode()).split(",")[0]
                self.file_exist_flag = str(msg.decode()).split(",")[-1]
                date = msg2.split("/")[-1]
                self.deviceid = str(msg.decode()).split("/")[-2]
                self.first_path = f"C:/Users/abc/Desktop/신종감염병/yolov5/files/{self.deviceid}"
                self.folder_path = f"C:/Users/abc/Desktop/신종감염병/yolov5/files/{self.deviceid}/{date}"
                self.input_path = f'{self.folder_path}/input'
                self.rgb_img = f'{self.input_path}/photo.jpg'
                self.ir_img = f'{self.input_path}/msx.jpg'
                self.save_dir = f'{self.input_path}'
                self.now = self.folder_path.split('/')[-1]

                executor = ThreadPoolExecutor(2)
                self.future2 = executor.submit(flyr.unpack, self.ir_img, )

                img0 = cv2.imread(self.rgb_img)
                img = letterbox(img0, self.img_size, stride=self.stride, auto=self.pt)[0]
                img = img.transpose((2, 0, 1))[::-1]
                im = np.ascontiguousarray(img)

                self.face_cnt = 0
                self.face_ids = []
                self.confidences = []
                self.bboxes = []
                self.labels = []
                self.temperatures = []
                self.class_name = []

                self.seen, self.windows, self.dt = 0, [], [0.0, 0.0, 0.0]

                t1 = time.perf_counter()
                im = torch.from_numpy(im)
                im = im.to(self.device)
                im = im.half() if self.model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]
                t2 = time.perf_counter()
                self.dt[0] += t2 - t1

                pred = self.model(im)
                t3 = time.perf_counter()
                self.dt[1] += t3 - t2

                #NMS
                torch.cuda.synchronize()
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
                self.dt[2] = time.perf_counter() - t3

                try:
                    imt = self.future2.result()
                except Exception:
                    print('flyr.py_error')
                # print('line191', imt)
                # imt = self.future2.result()


                for i, det in enumerate(pred):
                    self.seen += 1
                    p, im0 = self.rgb_img, img0.copy()
                    p = Path(p)
                    save_path = f'{self.save_dir}/{p.name}'
                    txt_path = f'{self.save_dir}/{p.name}'
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    imc = im0.copy() if self.save_crop else im0
                    self.imd = im0.copy() if self.debug_img else im0
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.name))
                    LOGGER.info(f"{'Class'.ljust(15)} {'Temperature'.ljust(20)} {'Confidence'.ljust(20)} {'BBOX'.ljust(30)}")
                    rgb_h, rgh_w = im0.shape[:2]
                    self.hr, self.wr = (self.irh/rgb_h), (self.irw/rgh_w)
                    if len(det):
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            w,h = xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                            w_ratio, h_ratio = w/h, h/w

                            if w_ratio >= 0.4 and h_ratio >= 0.4:
                                bbox = list(int(v.item()) for v in xyxy)
                                self.face_ids.append(f'face_{str(self.face_cnt).zfill(3)}')
                                self.confidences.append(conf)
                                self.bboxes.append(bbox)
                                self.labels.append(int(cls.item()))

                                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                xt1, yt1, xt2, yt2 = int(x1*self.wr) - 3, int(y1 * self.hr) + 18, int(x2 * self.wr) - 3, int(y2 * self.hr) + 18
                                xt1 = 0 if xt1 < 0 else xt1
                                yt2 = self.irh if yt2 > self.irh else yt2

                                # Distance
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                width, height = xywh[2:]
                                diag = (((width ** 2) + (height ** 2)) ** (1 / 2))
                                dist = ((-(math.log10(diag))) * 5) ** 3
                                distance = dist * (33.99 / dist + 2.0585) if dist <= 70.624 else dist * (64 / dist + 1.633574)

                                yt2 = 640 if yt2 > 640 else yt2
                                temp = imt.celsius[yt1:yt2, xt1:xt2].max()
                                temperature = 330 / (temp + 30) + 31.37144 if distance <= 272.182 else 10.5 / (
                                        temp - 265.6) + 36.40
                                self.temperatures.append(temperature)
                                self.class_name.append(self.mask_classes[int(cls.item())])
                                LOGGER.info(f"{colorstr('bold', self.mask_classes[int(cls.item())].ljust(15))} "
                                            f"{colorstr('bold', str(round(temperature, 2)).ljust(20))} "
                                            f"{colorstr('bold', str(round(conf.item(), 2)).ljust(20))} "
                                            f"{colorstr('bold', str(bbox).ljust(30))}")

                                # Write to file
                                if self.save_txt:
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized
                                    line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                                    # print('245line', line)
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colorss[c])
                                    if self.save_crop:
                                        save_one_box(xyxy, imc, file=f"{self.save_dir}/crops/{self.names[c]}/{p.stem}.jpg", BGR=True)
                                self.face_cnt += 1

                    with open(f'{self.input_path}/output.csv', 'w', newline='') as csv_out:
                        log = csv.writer(csv_out)
                        header = ['', 'face_id', 'class', 'temperature', 'confidence', 'bbox', 'event']
                        log.writerow(header)
                        index = 0

                        for face, label, temp, confidence, box in zip(self.face_ids, self.labels, self.temperatures, self.confidences, self.bboxes):
                            row = [index, face, self.mask_classes[label], round(temp, 3), round(confidence.item(), 3), box]

                            if label == 0 or label == 2 or temp > self.fever:
                                event_list = []
                                if label == 0 or label == 2:
                                    event_list.append(self.mask_classes[label])
                                if temp > self.fever:
                                    event_list.append('fever')
                                event = {'desc': ' '.join(str(s) for s in event_list),
                                         'bbox': ' '.join(str(s) for s in box),
                                         'temp': str(round(temp, 3))}

                                self.events.append(event)
                                row.append(event)

                            log.writerow(row)
                            index += 1

                        if self.send_event and self.events and self.file_exist_flag == 'False':
                            files = (f'{self.deviceid}_{self.now}.jpg', open(self.rgb_img, 'rb'), 'image/jpeg')
                            data = MultipartEncoder(fields={'deviceCd': self.deviceid,
                                                            'file': files,
                                                            'domainCd': 'BTP',
                                                            'data': json.dumps([v for v in self.events]),
                                                            'date': f'20{self.now[:2]}-{self.now[2:4]}-{self.now[4:6]} '
                                                                    f'{self.now[7:9]}:{self.now[9:11]}:{self.now[11:13]}',
                                                            'deviceGrpCd': 'feverCam'})
                            r = requests.post(self.thub_address, data=data, headers={'Content-Type': data.content_type})
                            print("devicecd : ", self.deviceid)
                            # print('line293thub', data)

                        if self.send_event and self.events and self.file_exist_flag == "False":
                            print(r.text)

                        if self.debug_img:
                            self.cv2_debug_img(self.ir_img, self.imd, self.labels, self.temperatures, self.confidences, self.bboxes, self.now, self.folder_path, self.wr, self.hr, self.temp_threshold,self.class_name)
                            # print('281_line_done')


                        t = tuple(x / self.seen * 1E3 for x in self.dt)
                        LOGGER.info(
                            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.img_size)}' % t)
                        if self.save_txt or self.save_img:
                            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
                        if self.update:
                            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)
                    # return combine_img

    def run_flask(self):
        flask_app.app.run(host="0.0.0.0", port=5000, debug=False)

    def multi_processing_flask_and_mask(self):
        total_list = []

        flask = ThreadPoolExecutor(2)
        flask.submit(self.run_flask)
        Processes1 = self.mp.Process(target=self.yolo_model_and_thub_send(), name='Processes1') #self.future2

        Processes1.daemon = True
        Processes1.start()

        for act_chiled in self.mp.active_children():
            total_list.append(act_chiled)

        while True:
            if not Processes1.is_alive():
                name =' Processes' + str(len(total_list)+1)
                total_list.append(name)
                Processes1 = self.mp.Process(target=self.yolo_model_and_thub_send(), name=name)
                Processes1.daemon = True
                Processes1.start()



def main():
    multi_rtsp = yolo_detect()
    multi_rtsp.multi_processing_flask_and_mask()

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method(method='spawn', force=True)
    main()
