import datetime
import os
import requests
import socket
import shutil
import time
# from datetime import datetime

from flask import request, Flask

app = Flask(__name__)

@app.route('/', methods=['POST'])
def tinker():
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # now = datetime.now()
    # date = now.microsecond

    SERVER_IP = '192.168.50.136'
    SERVER_PORT = 28283
    SIZE = 1024
    SERVER_ADDR = (SERVER_IP, SERVER_PORT)
    history_amount = 999

    files = request.files.getlist("file")
    deviceCd = request.values.get("deviceID")


    path = f"files/{deviceCd}/{date}"
    path_for_rm = f"files/{deviceCd}"

    input_path = f"{path}/input"
    output_path = f"{path}/output"

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        # start_time = time.time()
        file.save(os.path.join(input_path, file.filename))
        print(file.filename, f'saved in {date}')
        # print('end_time',time.time()-start_time)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        start_time = time.time()
        client_socket.connect(SERVER_ADDR)
        client_socket.send(path.encode())
        # print('########################')
        # print(client_socket.send(path.encode()))
        # print('end_time', time.time() - start_time)


        # msg = client_socket.recv(SIZE)

    while len(os.listdir(path_for_rm)) > history_amount:
        f = os.listdir(path_for_rm)
        f.sort()
        shutil.rmtree(rf'{path_for_rm}/{f[0]}')

    return "done"
    # return print('end_time', time.time() - start_time)


@app.route('/test', methods=['POST'])
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

@app.route('/test2', methods=['POST'])
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


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)  # Fowarded Address = http://1.209.33.170:22170/

