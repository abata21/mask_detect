import os
import datetime
import socket
from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import time
import shutil

app = Flask(__name__)
limiter = Limiter(app=app,key_func=get_remote_address, default_limits=["10 per second"])

@app.route('/', methods = ['GET','POST'])
@limiter.limit('6 per second')
def flask_get():
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    SERVER_IP = 'my_ip'
    SERVER_PORT = 28283
    SERVER_ADDR = (SERVER_IP, SERVER_PORT)
    history_amount = 9999

    detect_imgs = request.files.getlist('file')
    device_name = request.values.getlist('deviceID')[0]

    path = f"./{device_name}/{date}"
    input_path = f"{path}/input"
    os.makedirs(input_path, exist_ok=True)

    for img in detect_imgs:
        file_exist_flag = "False"
        if os.path.exists(os.path.join(input_path, img.filename)):
            file_exist_flag = "True"
        img.save(f'{input_path}/{secure_filename(img.filename)}')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            if not device_name == None:
                client_socket.connect(SERVER_ADDR)
                data = path + "," + file_exist_flag
                client_socket.send(data.encode())
            # time.sleep(0.1)
        except ConnectionRefusedError:
            print('Socket_error')

    return 'done'

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=False)
