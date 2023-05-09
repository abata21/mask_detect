import requests

url = 'http://disol.asuscomm.com:41878'
# url = 'http://1.209.33.170:41878'
# url = 'http://192.168.50.162:5000'
test_path = 'files/socket/test2/input'

files = [('file', open(f'{test_path}/photo.jpg', 'rb')), ('file', open(f'{test_path}/msx.jpg', 'rb'))]

device = 'test'

requests.post(url, files=files, data={"deviceID": device})

