import datetime
import os
import requests
import socket
import shutil
import time
# from datetime import datetime

from flask import request, Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(app=app,key_func=get_remote_address, default_limits=["10 per second"])

@app.route('/', methods=['GET','POST'])
@limiter.limit('9 per second')
# app = Flask(__name__)
#
# @app.route('/', methods=['GET','POST'])
# @limiter.limit('10 per seconds')
def tinker():
    # if 500000<int(request.headers['Content-Length']) < 1600000:
    # if len(request.headers['Content-Type']) > 60:
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # now = datetime.datetime.now()
    # date = now.microsecond

    SERVER_IP = '192.168.50.136'
    SERVER_PORT = 28283
    SIZE = 1024
    SERVER_ADDR = (SERVER_IP, SERVER_PORT)
    history_amount = 9999

#############################################################################
    # print(request)
    # print('길이', request.headers['Content-Length'])
    # print(request.headers)

#############################################################################

    # print('Type',request.headers['Content-Type'])
    # print('Type2',len(request.headers['Content-Type']))
    # print('deviceID', request.files.)

    files = request.files.getlist("file")
    print('files_ok')
    # print('files', len(request.files['file'].read()))
    # print(request.files.getlist)
    # print(files)
    deviceCd = request.values.get("deviceID")
    print('deviceID_ok')
    # print(deviceCd)
    # print(' ')
    # print('request', request)
    # print('request.header',request.headers)
    # print('files',files)
    # print('device_id',deviceCd)
    # print(' ')

    path = f"files/{deviceCd}/{date}"
    path_for_rm = f"files/{deviceCd}"

    input_path = f"{path}/input"
    output_path = f"{path}/output"

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        file_exist_flag = "False"
        if os.path.exists(os.path.join(input_path, file.filename)):
            file_exist_flag = "True"
        file.save(os.path.join(input_path, file.filename))
        print(file.filename, f'saved in {date}')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(SERVER_ADDR)
        # client_socket.send(path.encode())
        data = path + "," + file_exist_flag
        # print('data',data)
        client_socket.send(data.encode())
        print('send')
        # print(data.encode())
        # print(client_socket.send(path.encode()))
        # print('end_time', time.time() - start_time)

        # msg = client_socket.recv(SIZE)
        # print('size : ', msg)

    while len(os.listdir(path_for_rm)) > history_amount:
        f = os.listdir(path_for_rm)
        f.sort()
        shutil.rmtree(rf'{path_for_rm}/{f[0]}')

    print('one_cycle_end')
    print('                                     ')
    return "done"
# return print('end_time', time.time() - start_time)


@app.route('/test', methods=['GET','POST'])
def just_save():
    files = request.files.getlist("file")
    deviceCd = request.values.get("deviceID")
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    path = f"files/{deviceCd}/{date}"
    input_path = f"{path}/input"
    output_path = f"{path}/output"

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        file.save(os.path.join(input_path, file.filename))
        print(file.filename, f'saved in {date}')

    return "done"

@app.route('/test2', methods=['GET','POST'])
def just_save_img():
    files = request.files.getlist("file")
    deviceCd = request.values.get("deviceID")
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    path = f"files/{deviceCd}"

    os.makedirs(f'{path}/thermal', exist_ok=True)
    os.makedirs(f'{path}/rgb', exist_ok=True)
    for file in files:
        fp = f'{path}/thermal' if file.filename == "msx.jpg" else f'{path}/rgb'
        file.save(os.path.join(fp, f'{date}.jpg'))


    return "done"
#

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # Fowarded Address = http://1.209.33.170:22170/

