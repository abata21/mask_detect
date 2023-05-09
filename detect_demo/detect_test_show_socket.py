import os
import shutil
import socket
import datetime
import time

SERVER_IP = '192.168.50.162'
SERVER_PORT = 28282
SIZE = 1024
SERVER_ADDR = (SERVER_IP, SERVER_PORT)
paths = ['files/socket/test2/input', 'files/socket/test/input']
while True:
    for p in paths:
        print(p)
        date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        path = f'files/show'
        rgb_path = f'{path}/rgb'
        ir_path = f'{path}/ir'
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(ir_path, exist_ok=True)

        shutil.copy(f'{p}/photo.jpg', f'{rgb_path}/{date}.jpg')
        shutil.copy(f'{p}/msx.jpg', f'{ir_path}/{date}.jpg')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(SERVER_ADDR)
            client_socket.send(date.encode())
            time.sleep(1)
