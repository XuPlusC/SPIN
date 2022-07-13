from paho.mqtt import client as mqtt_client
import re
import numpy as np
import cv2
import base64
import datetime
import json
import requests
import pose_detect
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')


def addCam():
    # url_http = "http://juntai.vip3gz.91tunnel.com/pose/free/alarm/pump"
    url_http = "http://127.0.0.1:8189/free/device"
    headers = {'Content-Type': 'application/json'}
    ISOTIMEFORMAT = "%Y-%m-%d %H:%M:%S"

    # print(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    # send fall down alert to server
    payload = {
        "type": "object",
        "id": "testCameraId1",
        "house": "东二楼-2单元703",
        "village": "东二区",
        "address": "华中科技大学",
        "lon": "30.51987991632215",
        "lat": "114.41435224713373",
        "household": "黄鸳莎",
        "phoneNumber": "1253615316",
    }
    r = requests.post(url_http, headers=headers, data=json.dumps(payload))
    print(r.status_code)
    # print(r.content)


class MQTTClient(mqtt_client.Client):
    def __init__(self, checkpoint):
        super(MQTTClient, self).__init__()
        self.poseDetector = pose_detect.PoseDetect(checkpoint)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe("marsaii/#")    # 订阅$SYS/下的所有主题
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_message(client, userdata, msg):
        # print(msg.topic + " " + str(msg.payload)[0:40] + " ....")
        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # matchRes = re.findall(b'\"image\":\"(.*)\",\"label\"', msg.payload)
        # matchRes = re.findall(b'\"image\":\"(.*)\",\"ppi\"', msg.payload)

        # if len(matchRes) != 0:
        #     # print("successfully found base64 image!") 
        #     # list_splitRes = matchRes[0].split('\\r\\n')
        #     # print("matchRes:", len(matchRes))
        #     # print("split result:", len(list_splitRes))

        #     # strNoNewLine = re.sub(b'\\r\\n', '', matchRes[0])
        #     strNoNewLine = matchRes[0]
        #     strNoNewLine = strNoNewLine.replace(b'\\r\\n', b'')
        #     missing_padding = 4 - len(strNoNewLine) % 4
        #     if missing_padding:
        #         strNoNewLine += b'=' * missing_padding
        #     imageData_bytes = base64.b64decode(strNoNewLine)
        #     imageData_nparr = np.fromstring(imageData_bytes, np.uint8)
        #     image = cv2.imdecode(imageData_nparr, -1)
        #     cv2.imshow("wee", image)
        #     cv2.waitKey(1)
        #     # client.poseDetector.processFrame(image)

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
        
        # cv2.imshow('cam1', img)
        client.poseDetector.processFrame(img)

        key = cv2.waitKey(10)
        if key == 113:
            exit()


if __name__ == "__main__":
    args = parser.parse_args()

    cv2.namedWindow("cam1", 0)

    client = MQTTClient(args.checkpoint)
    client.on_connect = MQTTClient.on_connect    # 连接broker时broker响应的回调
    client.on_message = MQTTClient.on_message    # 接收到订阅消息时的回调

    client.connect("192.168.1.3", 1883, 60)    # 连接到broker

    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
    client.loop_forever()    # 保持永久连接
