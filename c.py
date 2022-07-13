import base64
import cv2
import json
import numpy as np
import os
import time
from paho.mqtt import client as mqtt_client

# with open(os.path.join(os.path.split(__file__)[0], 'config.json')) as f:
#     conf = json.loads(f.read())


conf = {
    'topic': 'marsaii',
    'ip': '192.168.1.3',
    'port': '1883',
    'username': 'test',
    'password': '123456',
}


def connect():
    def on_connect(cli, data, flags, rc):
        if rc == 0:
            cli.subscribe(conf['topic'])
            return
        raise Exception(f'MQTT连接失败,状态码->{rc}')

    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.on_message = handle_msg
    client.username_pw_set(username=conf['username'], password=conf['password'])
    client.connect(conf['ip'], int(conf['port']))
    print('MQTT连接成功!')
    return client


def handle_msg(cli, _, msg):
    print(f'[{time.ctime()}]收到消息!!!')
    try:
        res = json.loads(msg.payload)
        assert isinstance(res, dict)
        assert not ({'image', 'label', 'posttime', 'source'} - res.keys())
    except (json.JSONDecodeError, AssertionError) as e:
        print(f'消息无法解析, 已丢弃->{e}')
        return

    image_string = res['image']
    np_arr = np.frombuffer(base64.b64decode(image_string), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # for item in res['label']['data']:
    #     cv2.rectangle(img, (item['x1'], item['y1']), (item['x2'], item['y2']), (0, 0, 0), thickness=3)
    cv2.imshow('img', img)
    key = cv2.waitKey(10)
    if key == 113:
        exit()


if __name__ == '__main__':
    client = connect()
    client.loop_forever()
