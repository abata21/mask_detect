import socket
import cv2
import base64
import numpy
from _thread import *

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def threaded(client_socket, addr):
    print('>> Connected by :', addr[0], ':', addr[1])
    while True:
        try:
            recvedLength = recvall(client_socket, 64)
            dataLength = recvedLength.decode('utf-8')
            stringData = recvall(client_socket, int(dataLength))
            data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
            decimg = cv2.imdecode(data, 1)
            cv2.imshow("Test", decimg)
            cv2.waitKey(1)

        except ConnectionResetError as e:
            print('>> Disconnected by ' + addr[0], ':', addr[1])
            break

    client_socket.close()

def main():
    HOST = '127.0.0.1'
    PORT = 9999

    print('>> Server Start')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    try:
        while True:
            client_socket, addr = server_socket.accept()
            start_new_thread(threaded, (client_socket, addr))
    except Exception as e:
        print('에러 : ', e)
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()