import os
import datetime
import socket
from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
limiter = Limiter(app=app,key_func=get_remote_address, default_limits=["10 per second"])

@app.route('/', methods = ['GET','POST'])
@limiter.limit('7 per second')
def flask_get():
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    SERVER_IP = '192.168.50.136'
    SERVER_PORT = 28283
    SERVER_ADDR = (SERVER_IP, SERVER_PORT)
    history_amount = 9999

    detect_imgs = request.files.getlist('file')
    device_name = request.values.getlist('deviceID')[0]

    path = f"C:/Users/abc/Desktop/신종감염병/yolov5/files/hyo/{device_name}/{date}"
    path_for_rm = f"C:/Users/abc/Desktop/신종감염병/yolov5/files/hyo/{device_name}"
    input_path = f"{path}/input"
    os.makedirs(input_path, exist_ok=True)

    for img in detect_imgs:
        file_exist_flag = "False"
        if os.path.exists(os.path.join(input_path, img.filename)):
            file_exist_flag = "True"
        img.save(f'{input_path}/{secure_filename(img.filename)}')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect(SERVER_ADDR)
            data = path + "," + file_exist_flag
            client_socket.send(data.encode())
        except ConnectionRefusedError:
            print('연결문제')
    while len(os.listdir(path_for_rm)) > history_amount:
        f = os.listdir(path_for_rm)
        f.sort()
        shutil.rmtree(rf'{path_for_rm}/{f[0]}')
    return 'done'

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=False)