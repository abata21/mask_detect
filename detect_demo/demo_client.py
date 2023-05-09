import socket
import cv2
import base64
import numpy
import time

def main():
    HOST = '127.0.0.1'
    PORT = 9999
    count = 0
    filename = "z.jpg"

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    while True:
        if count % 2 == 1:
            filename = "z.jpg"
        else:
            filename = "a.jpg"
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        resize_frame = cv2.resize(img, dsize=(480, 640), interpolation=cv2.INTER_AREA)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
        print(result)

        data = numpy.array(imgencode)
        stringData = base64.b64encode(data)
        length = str(len(stringData))

        client_socket.sendall(length.encode('utf-8').ljust(64))
        client_socket.send(stringData)
        count += 1

        time.sleep(1)

    client_socket.close()


if __name__ == "__main__":
    main()