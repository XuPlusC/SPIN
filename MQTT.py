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


# MQTT connection
# by cococat 2022.3.17
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe("marsaii/#")    # 订阅$SYS/下的所有主题
    else:
        print("Failed to connect, return code %d\n", rc)


def on_message(client, userdata, msg, poseDetector):
    print(msg.topic+" "+str(msg.payload))
    print(str(msg.payload, 'utf-8'))

    # matchRes = re.findall('\"image\":\"(.*)\",\"ppi\"', str(msg.payload, 'utf-8'))
    matchRes = re.findall(b'\"image\":\"(.*)\",\"ppi\"', msg.payload)

    if len(matchRes) != 0:
        strNoNewLine = matchRes[0]
        strNoNewLine = strNoNewLine.replace(b'\\r\\n', b'')
        missing_padding = 4 - len(strNoNewLine) % 4
        if missing_padding:
            strNoNewLine += b'=' * missing_padding
        imageData_bytes = base64.b64decode(strNoNewLine)
        imageData_nparr = np.fromstring(imageData_bytes, np.uint8)
        image = cv2.imdecode(imageData_nparr, -1)
        poseDetector.processFrame(image)
        # file = open('1.jpg', 'wb')
        # file.write(imageData_bytes)
        # file.close()


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
            client.subscribe("test/#")    # 订阅$SYS/下的所有主题
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_message(client, userdata, msg):
        # print(msg.topic+" "+str(msg.payload))
        # print(str(msg.payload, 'utf-8'))

        # matchRes = re.findall('\"image\":\"(.*)\",\"ppi\"', str(msg.payload, 'utf-8'))
        # matchRes = re.findall(b'\"image\":\"(.*)\",\"ppi\"', msg.payload)
        # matchRes = re.findall(b'\"image\" : \"(.*)\",\r\n\"label\" :', msg.payload)
        matchRes = re.findall(b'\"image\" : \"(.*)\",', msg.payload)
        # ,"label":{"data":
        

        if len(matchRes) != 0:
            print(msg.topic)
            # list_splitRes = matchRes[0].split('\\r\\n')
            # print("matchRes:", len(matchRes))
            # print("split result:", len(list_splitRes))

            # strNoNewLine = re.sub(b'\\r\\n', '', matchRes[0])
            strNoNewLine = matchRes[0]
            strNoNewLine = strNoNewLine.replace(b'\\r\\n', b'')
            missing_padding = 4 - len(strNoNewLine) % 4
            if missing_padding:
                strNoNewLine += b'=' * missing_padding

            imageData_bytes = base64.b64decode(strNoNewLine)
            # filename = "F:/Pose/Workspace/Python/SPIN/wee.png"
            # with open(filename, "wb") as f:
            #     f.write(imageData_bytes)

            imageData_nparr = np.fromstring(imageData_bytes, np.uint8)
            image = cv2.imdecode(imageData_nparr, 1)
            # cv2.imshow("cam1", image)
            # cv2.waitKey(1)
            client.poseDetector.processFrame(image)


if __name__ == "__main__":
    args = parser.parse_args()
    
    # poseDetector = pose_detect.PoseDetect(args.checkpoint)

    # client = mqtt_client.Client()
    # client.on_connect = on_connect    # 连接broker时broker响应的回调
    # client.on_message = on_message    # 接收到订阅消息时的回调

    # client.connect("192.168.1.100", 1883, 60)    # 连接到broker

    # # Blocking call that processes network traffic, dispatches callbacks and
    # # handles reconnecting.
    # # Other loop*() functions are available that give a threaded interface and a
    # # manual interface.
    # client.loop_forever()    # 保持永久连接

    client = MQTTClient(args.checkpoint)
    client.on_connect = MQTTClient.on_connect    # 连接broker时broker响应的回调
    client.on_message = MQTTClient.on_message    # 接收到订阅消息时的回调

    client.connect("192.168.0.137", 1883, 60)    # 连接到broker

    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
    client.loop_forever()    # 保持永久连接
