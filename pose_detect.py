from turtle import Turtle
import torchvision
import cv2
import torch
import numpy as np

import argparse
from models import hmr, SMPL
import config
import constants
import datetime
import frame_pose
from paho.mqtt import client as mqtt_client
import base64
import json
import math
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=None, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')


def plot_box(img, box, box_type="xywh", color=(255, 0, 0)):
    """在图片上绘制矩形框"""

    if box_type == "xywh":
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
    elif box_type == "xyxy":
        x1, y1, x2, y2 = box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)


def imshow(title, img, wait_ms=10):
    """显示图像 Esc键退出"""

    if wait_ms <= 0:
        while True:
            cv2.imshow(title, img)
            if cv2.waitKey(20) == 27:  # Esc
                break
    else:
        cv2.imshow(title, img)
        cv2.waitKey(wait_ms)


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [
        COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]["labels"].cpu().numpy())
    ]  # Get the Prediction Score
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].detach().cpu().numpy())
    ]  # Bounding boxes
    pred_score = list(pred[0]["scores"].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_classes = pred_classes[: pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == "person":
            person_boxes.append(box)

    return person_boxes


def readMatrixFromFile(filepath):
    # ret = np.array([])
    # with open(filepath, 'r') as f:
    #     my_data = f.readlines()  # txt中所有字符串读入data，得到的是一个list
    #     # 对list中的数据做分隔和类型转换
    #     for line in my_data:
    #         line_data = line.split()
    #         ret += np.array(line_data)
    ret = np.loadtxt(filepath)
    return ret


def do_mosaic(frame, x, y, w, h, neighbor=8):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x :  打码区域左顶点x
    :param int y:  打码区域左顶点y
    :param int w:  打码区域宽
    :param int h:  打码区域高
    :param int neighbor:  马赛克每一块的宽
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        print("超出视频尺寸！")
        return
    for i in range(0, h - neighbor, neighbor):
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            # color=frame[262][967].tolist()
            color = frame[i + y][j + x].tolist()
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)


class PoseDetect:
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)
    timeFormat_ISO = "%Y-%m-%d %H:%M:%S"
    timeFormat_File = "%Y-%m-%d_%H:%M:%S"

    # url_http = "http://juntai.vip3gz.91tunnel.com/pose/free/alarm/pump"
    # url_http = "http://127.0.0.1:8189/free/alarm/pump"
    url_http = "http://47.104.74.43:8189/free/alarm/pump"
    headers = {'Content-Type': 'application/json'}

    def __init__(self, pCheckpoint) -> None:
        PoseDetect.box_model.cuda()
        PoseDetect.box_model.eval()

        # SMPL : Load pretrained model
        checkpoint = torch.load(pCheckpoint)
        PoseDetect.model.load_state_dict(checkpoint['model'], strict=False)
        PoseDetect.model.eval()

        # TODO: cap!
        # input_width, input_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fall_down_frame_counter = 0
        self.fall_down_frame_threshold = 4
        self.fall_down_list = []
        self.fall_down_target_counter = 0
        self.b_send_fall_down_alert = False  # only send alert when new fall down appears. do not keep sending when person has already fallen on the ground

        self.image_showcase = 0
        self.origin_image = 0
        self.skeleton_image = 0

        # # skeleton on gray background. by cococat 2022.2.28
        # img_black = np.zeros((input_height, input_width, 3), np.uint8)
        # img_black.fill(55)

        # # save human enterance & exit info to file
        # # by cococat 2022.3.7

        # self.b_person_in_room = False

        # b_available_cap_exsist = True

        # # record human movement and stasis
        # list_pelvis_queue = [[] for n in range(len(list_cap))]      # list，记录人体中心点的队列
        # list_b_person_movement_queue = [[] for n in range(len(list_cap))]  # list，记录每个cap中的人是否在当前帧移动的队列
        # stasis_threshold_x = input_width / 100
        # stasis_threshold_y = input_height / 125

        # # send fall info via http request
        # # by cococat 2022.3.21
        # # url_http = "127.0.0.1:8189/free/alarm/pump"

        print("weee")

    def processFrame(self, image_bgr):
        input = []
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().cuda()
        input.append(img_tensor)

        box_list = get_person_detection_boxes(PoseDetect.box_model, input, threshold=0.9)
        square_box_list = []
        for box in box_list:
            (x1, y1), (x2, y2) = box
            box = [x1, y1, x2, y2]

            box_height = y2 - y1
            box_width = x2 - x1
            box_center = np.array([x1 + box_width / 2, y1 + box_height / 2])
            square_width = max(box_height, box_width)
            square_width *= 1.05
            new_upper_left_x = box_center[0] - square_width / 2
            new_upper_left_y = box_center[1] - square_width / 2
            new_lower_right_x = box_center[0] + square_width / 2
            new_lower_right_y = box_center[1] + square_width / 2
            square_box = [(new_upper_left_x if (new_upper_left_x > 0) else 0, new_upper_left_y if (new_upper_left_y > 0) else 0),
                            (new_lower_right_x if (new_lower_right_x <= image_bgr.shape[1]) else image_bgr.shape[1], new_lower_right_y if (new_lower_right_y <= image_bgr.shape[0]) else image_bgr.shape[0])]
            square_box_list.append(square_box)

        # 剔除重合的square_box
        new_square_box_list = []
        intersection_rate_thre = 0.9
        for i in range(0, len(square_box_list)):
            rect1 = square_box_list[i]
            coincidence = False
            for j in range(0, len(square_box_list)):
                if(i == j):
                    continue

                rect2 = square_box_list[j]
                x1 = max(rect1[0][0], rect2[0][0])
                y1 = max(rect1[0][1], rect2[0][1])
                x2 = min(rect1[1][0], rect2[1][0])
                y2 = min(rect1[1][1], rect2[1][1])
                if (x1 < x2 and y1 < y2):
                    intersection_area = (x2 - x1) * (y2 - y1)
                    rect1_area = (rect1[1][0] - rect1[0][0]) * (rect1[1][1] - rect1[0][1])
                    print('intersection_rate is ', intersection_area / rect1_area)
                    if(intersection_area / rect1_area > intersection_rate_thre):
                        print('box coincidence')
                        coincidence = True
                        break
            if not coincidence:
                new_square_box_list.append(rect1)
        square_box_list = new_square_box_list

        if len(box_list) != 0:
            # Preprocess input image and generate predictions
            norm_img, bbox_centers, bbox_scales = frame_pose.process_image(img, square_box_list, input_res=constants.IMG_RES)
            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = PoseDetect.model(norm_img.to(PoseDetect.device))
                pred_output = PoseDetect.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
                # pred_vertices = pred_output.vertices

            # Calculate camera parameters for rendering
            camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:, 0] + 1e-9)], dim=-1)

            # project 3D joints to 2D image
            # TODO: batch size fix
            batch_size = 1
            SPIN_camera_joints = pred_output.joints
            img_joints = frame_pose.perspective_projection(SPIN_camera_joints,
                                                rotation=torch.eye(3, device=PoseDetect.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                translation=camera_translation,
                                                focal_length=constants.FOCAL_LENGTH,
                                                camera_center=torch.zeros(batch_size, 2, device=PoseDetect.device))

            # relocate and rescale 2D joints coordinate
            img_joints = img_joints.cpu().numpy()
            adjusted_joints_coord = frame_pose.adjust_joints_2D_coordinate(img_joints, bbox_centers, bbox_scales)

            img_black = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 3), np.uint8)
            img_black.fill(55)
            final_image = frame_pose.draw_points_and_skeleton(img_black, adjusted_joints_coord, frame_pose.skeleton)
            # final_image = frame_pose.draw_points_and_skeleton(image_bgr, adjusted_joints_coord, frame_pose.skeleton)
            self.skeleton_image = final_image
            self.origin_image = image_bgr
            final_image = np.concatenate((image_bgr, final_image), axis=1)
            self.image_showcase = final_image

            # # compare stored pelvis coordinate
            # current_pelvis = adjusted_joints_coord[0][8]
            # length_pelvis_queue = len(list_pelvis_queue[index_cap])
            # if length_pelvis_queue != 0:
            #     average_stored_pelvis_x = 0
            #     average_stored_pelvis_y = 0
            #     # b_temp_movement = false
            #     for index_stored_pelvis in range(length_pelvis_queue):
            #         average_stored_pelvis_x += list_pelvis_queue[index_cap][index_stored_pelvis][0]
            #         average_stored_pelvis_y += list_pelvis_queue[index_cap][index_stored_pelvis][1]
            #     average_stored_pelvis_x /= length_pelvis_queue
            #     average_stored_pelvis_y /= length_pelvis_queue
            #     if abs(current_pelvis[0] - average_stored_pelvis_x) < stasis_threshold_x and abs(current_pelvis[1] - average_stored_pelvis_y) < stasis_threshold_y:
            #         list_b_person_movement_queue[index_cap].append(False)
            #         # print("room ", str(index_cap), " static!")
            #     else:
            #         list_b_person_movement_queue[index_cap].append(True)
            #         # print("room ", str(index_cap), " moving!")

            #     # b_movement = False
            #     # for index_stored_pelvis in range(length_pelvis_queue):
            #     #     stored_frame_pelvis_x = list_pelvis_queue[index_cap][index_stored_pelvis][0]
            #     #     stored_frame_pelvis_y = list_pelvis_queue[index_cap][index_stored_pelvis][1]
            #     #     if abs(current_pelvis[0] - stored_frame_pelvis_x) > stasis_threshold_x and abs(current_pelvis[1] - stored_frame_pelvis_y) > stasis_threshold_y:
            #     #         b_movement = True
            #     #         break
            #     # if b_movement:
            #     #     list_b_person_movement_queue[index_cap].append(True)
            #     # else:
            #     #     list_b_person_movement_queue[index_cap].append(False)

            # else:
            #     list_b_person_movement_queue[index_cap].append(True)

            # if length_pelvis_queue == 5:
            #     list_pelvis_queue[index_cap].pop(0)
            # list_pelvis_queue[index_cap].append(current_pelvis)
            # if len(list_b_person_movement_queue[index_cap]) >= 8:
            #     list_b_person_movement_queue[index_cap].pop(0)

            # fall-down check:
            fall_down_list = frame_pose.fall_down_check(adjusted_joints_coord, image_bgr.shape[0])
            fallen_count = len(fall_down_list)
            if(fallen_count > 0):
                self.fall_down_frame_counter += 1
                if(fallen_count > self.fall_down_target_counter and self.fall_down_frame_counter > self.fall_down_frame_threshold):
                    self.fall_down_target_counter = fallen_count
                    self.b_send_fall_down_alert = True
                    # print("need to alert")
                else:
                    self.b_send_fall_down_alert = False
                # print("fall!")
            else:
                self.fall_down_frame_counter = 0
                self.b_send_fall_down_alert = False
                self.fall_down_target_counter = 0

            # if list_b_person_in_room[index_cap] is False:
            #     list_b_person_in_room[index_cap] = True
            #     str_log = datetime.datetime.now().strftime(timeFormat_ISO) + ": enter room " + str(index_cap) + " \n"
            #     print(str_log)
            #     with open('log.txt', 'a') as file_log_entrance_and_exit:
            #         file_log_entrance_and_exit.write(str_log)

        else:   # no person found in the frame
            pass
            # final_image = np.concatenate((image_bgr, img_black.copy()), axis=1)
            # # final_image = image_bgr

            # if list_b_person_in_room[index_cap] is True:
            #     list_b_person_in_room[index_cap] = False
            #     str_log = datetime.datetime.now().strftime(timeFormat_ISO) + ": leave room " + str(index_cap) + " \n"
            #     print(str_log)
            #     with open('log.txt', 'a') as file_log_entrance_and_exit:
            #         file_log_entrance_and_exit.write(str_log)

            # # clear the record queue
            # if(len(list_pelvis_queue[index_cap]) != 0):
            #     list_pelvis_queue[index_cap].clear()
            # if(len(list_b_person_movement_queue[index_cap]) != 0):
            #     print("clear person movement queue")
            #     list_b_person_movement_queue[index_cap].clear()

        # decide whether the person has fallen down or not
        # print bounding box on image
        for box_index, box in enumerate(square_box_list):
            (x1, y1), (x2, y2) = box
            box = [x1, y1, x2, y2]
            if self.fall_down_frame_counter > self.fall_down_frame_threshold and self.b_send_fall_down_alert is True:
                # # save file to local server approach
                # # temporarily disabled 2022.3.29
                # str_dateTimeFile = datetime.datetime.now().strftime(PoseDetect.timeFormat_File)
                # str_originImageFilePath = "/home/juntai/docker_apps/nginx/volumes/html/pose_images/origin/" + str_dateTimeFile + ".jpg"
                # str_skeletonImageFilePath = "/home/juntai/docker_apps/nginx/volumes/html/pose_images/pose/" + str_dateTimeFile + ".jpg"
                # cv2.imwrite(str_originImageFilePath, self.origin_image)
                # cv2.imwrite(str_skeletonImageFilePath, self.skeleton_image)

                # # send fall down alert to aliyun server
                # payload = {
                #     "id": "dong2CameraId1",
                #     "cameraImageUrl": str_dateTimeFile + ".jpg",
                #     "poseImageUrl": str_dateTimeFile + ".jpg",
                #     "type": 1,
                #     "alarmTime": datetime.datetime.now().strftime(PoseDetect.timeFormat_ISO)
                # }
                # r = requests.post(PoseDetect.url_http, headers=PoseDetect.headers, data=json.dumps(payload))
                # print(r.status_code)
                # print(r.content)

                # save file to aliyun server approach
                # by cococat 2022.3.29
                # str_dateTimeFile = datetime.datetime.now().strftime(PoseDetect.timeFormat_File)
                # str_originImageFilePath = "/home/juntai/docker_apps/nginx/volumes/html/pose_images/origin/" + str_dateTimeFile + ".jpg"
                # str_skeletonImageFilePath = "/home/juntai/docker_apps/nginx/volumes/html/pose_images/pose/" + str_dateTimeFile + ".jpg"
                # cv2.imwrite(str_originImageFilePath, self.origin_image)
                # cv2.imwrite(str_skeletonImageFilePath, self.skeleton_image)

                encoded_origin_image = cv2.imencode('.jpg', self.origin_image)[1]
                encoded_skeleton_image = cv2.imencode('.jpg', self.skeleton_image)[1]
                # strdata_origin_image = np.array(encoded_origin_image).tostring()
                # strdata_skeleton_image = np.array(encoded_skeleton_image).tostring()
                strdata_origin_image = str(base64.b64encode(encoded_origin_image))[2:-1]
                strdata_skeleton_image = str(base64.b64encode(encoded_skeleton_image))[2:-1]


                # send fall down alert to server
                payload = {
                    "id": "Xi6CameraId1",
                    "cameraImageUrl": strdata_origin_image,
                    "poseImageUrl": strdata_skeleton_image,
                    "type": 1,
                    "alarmTime": datetime.datetime.now().strftime(PoseDetect.timeFormat_ISO)
                }
                # print(payload)
                r = requests.post(PoseDetect.url_http, headers=PoseDetect.headers, data=json.dumps(payload))
                print(r.status_code)
                print(r.content)


                color = (255, 0, 0) if (fall_down_list.count(box_index) != 0) else (255, 0, 0)
                # color = (0, 0, 255) if (fall_down_list.count(box_index) != 0) else (255, 0, 0)
            else:
                color = (255, 0, 0)
            plot_box(self.image_showcase, box, "xyxy", color)

        # # print moving or static on image
        # if b_person_in_room is False:
        #     cv2.putText(final_image, "nobody", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 100, 255), 5)
        # else:
        #     move_frame_count = 0
        #     for index_list_b_move in range(len(list_b_person_movement_queue[index_cap])):
        #         if list_b_person_movement_queue[index_cap][index_list_b_move] is True:
        #             move_frame_count += 1

        #     if move_frame_count >= len(list_b_person_movement_queue[index_cap]) * 0.6:
        #         cv2.putText(final_image, "moving", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 100, 255), 5)
        #     else:
        #         cv2.putText(final_image, "static", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 100, 255), 5)

        cv2.imshow("cam1", self.image_showcase)
        cv2.waitKey(1)


if __name__ == "__main__":
    args = parser.parse_args()

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.cuda()
    box_model.eval()

    # SMPL : Load pretrained model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # SMPL: Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # cap = cv2.VideoCapture("rtsp://cococat:f51e7e7951598caf878fb26dc67d58cf@192.168.162.95/64")
    # cap = cv2.VideoCapture("/home/pose/Desktop/VID_20220226_153226.mp4")
    # cap = cv2.VideoCapture("/home/pose/Videos/horizon.mp4")
    # cap = cv2.VideoCapture("/home/pose/Videos/short.mp4")
    # cap = cv2.VideoCapture("/home/pose/Desktop/showcase_fall_mosaic.mp4")
    # cap = cv2.VideoCapture("/home/pose/Desktop/跌到视频（马赛克）.mp4")

    # cap1 = cv2.VideoCapture("rtsp://cococat:f51e7e7951598caf878fb26dc67d58cf@192.168.162.95/64")
    # cap2 = cv2.VideoCapture("rtsp://cococat:f51e7e7951598caf878fb26dc67d58cf@192.168.162.95/65")
    # cap3 = cv2.VideoCapture("rtsp://cococat:f51e7e7951598caf878fb26dc67d58cf@192.168.162.95/66")

    # cap1 = cv2.VideoCapture("/home/pose/Desktop/VID_20220226_153226.mp4")
    # cap2 = cv2.VideoCapture("/home/pose/Videos/horizon.mp4")
    # cap3 = cv2.VideoCapture("/home/pose/Videos/short.mp4")
    # list_cap = [cap, cap1, cap2, cap3]

    # cap = cv2.VideoCapture("/home/pose/Videos/3_12_static/64_horizon.mp4")
    # cap = cv2.VideoCapture("/home/pose/Videos/3_12_static/new67.mp4")
    # cap = cv2.VideoCapture("rtsp://192.168.1.201:8554/ds-test")
    # cap = cv2.VideoCapture("http://192.168.1.201:8080/proxy/0.flv")
    # cap = cv2.VideoCapture("/home/pose/Videos/horizon.mp4")
    cap = cv2.VideoCapture("rtsp://cococat:f51e7e7951598caf878fb26dc67d58cf@192.168.162.95/64")

    list_cap = [cap]

    input_width, input_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame extraction
    interval = 4  # 3
    counter = 0

    # save result as file
    input_FPS = cap.get((cv2.CAP_PROP_FPS))
    extracted_FPS = input_FPS / interval
    fourCC = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter("/home/pose/Videos/debug.mp4", fourCC, extracted_FPS, (2000, 500))

    fall_down_frame_counter = 0
    fall_down_frame_threshold = extracted_FPS / 2  # falled down for 0.5 second
    fall_down_list = []
    list_fall_down_frame_counter = [0 for n in range(len(list_cap))]

    # skeleton on gray background. by cococat 2022.2.28
    img_black = np.zeros((input_height, input_width, 3), np.uint8)
    img_black.fill(55)

    # save human enterance & exit info to file
    # by cococat 2022.3.7
    ISOTIMEFORMAT = "%Y-%m-%d %H:%M:%S"
    list_b_person_in_room = [False for n in range(len(list_cap))]

    b_available_cap_exsist = True

    # record human movement and stasis
    list_pelvis_queue = [[] for n in range(len(list_cap))]      # list，记录人体中心点的队列
    list_b_person_movement_queue = [[] for n in range(len(list_cap))]  # list，记录每个cap中的人是否在当前帧移动的队列
    stasis_threshold_x = input_width / 100
    stasis_threshold_y = input_height / 125

    # send fall info via http request
    # by cococat 2022.3.21
    # url_http = "127.0.0.1:8189/free/alarm/pump"
    url_http = "http://127.0.0.1:8189/pose/free/alarm/pump"
    headers = {'Content-Type': 'application/json'}

    while (b_available_cap_exsist):
        b_available_cap_exsist = False
        list_final_image = []
        for index_cap in range(len(list_cap)):
            temp_cap = list_cap[index_cap]
            if (temp_cap.isOpened()):
                ret, image_bgr = temp_cap.read()
                if(ret is not True):
                    continue

                b_available_cap_exsist = True

                if interval == 0 or counter % interval == 0:
                    input = []
                    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().cuda()
                    input.append(img_tensor)

                    box_list = get_person_detection_boxes(box_model, input, threshold=0.9)

                    square_box_list = []
                    for box in box_list:
                        (x1, y1), (x2, y2) = box
                        box = [x1, y1, x2, y2]

                        box_height = y2 - y1
                        box_width = x2 - x1
                        box_center = np.array([x1 + box_width / 2, y1 + box_height / 2])
                        square_width = max(box_height, box_width)
                        square_width *= 1.05
                        new_upper_left_x = box_center[0] - square_width / 2
                        new_upper_left_y = box_center[1] - square_width / 2
                        new_lower_right_x = box_center[0] + square_width / 2
                        new_lower_right_y = box_center[1] + square_width / 2
                        square_box = [(new_upper_left_x if (new_upper_left_x > 0) else 0, new_upper_left_y if (new_upper_left_y > 0) else 0),
                                        (new_lower_right_x if (new_lower_right_x <= image_bgr.shape[1]) else image_bgr.shape[1], new_lower_right_y if (new_lower_right_y <= image_bgr.shape[0]) else image_bgr.shape[0])]
                        square_box_list.append(square_box)

                    if len(box_list) != 0:
                        # Preprocess input image and generate predictions
                        norm_img, bbox_centers, bbox_scales = frame_pose.process_image(img, square_box_list, input_res=constants.IMG_RES)
                        with torch.no_grad():
                            pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
                            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
                            pred_vertices = pred_output.vertices

                        # Calculate camera parameters for rendering
                        camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:, 0] + 1e-9)], dim=-1)

                        # project 3D joints to 2D image
                        # TODO: batch size fix
                        batch_size = 1
                        SPIN_camera_joints = pred_output.joints
                        img_joints = frame_pose.perspective_projection(SPIN_camera_joints,
                                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                                            # rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                            translation=camera_translation,
                                                            focal_length=constants.FOCAL_LENGTH,
                                                            camera_center=torch.zeros(batch_size, 2, device=device))

                        # relocate and rescale 2D joints coordinate
                        img_joints = img_joints.cpu().numpy()
                        adjusted_joints_coord = frame_pose.adjust_joints_2D_coordinate(img_joints, bbox_centers, bbox_scales)

                        # draw 2D joints on frame
                        # final_image = frame_pose.draw_points_and_skeleton(image_bgr, adjusted_joints_coord, frame_pose.skeleton)

                        final_image = frame_pose.draw_points_and_skeleton(img_black.copy(), adjusted_joints_coord, frame_pose.skeleton)
                        cv2.imwrite("/home/pose/Workspace/Python/Test/FallImage/Pose/test.jpg", final_image)
                        cv2.imwrite("/home/pose/Workspace/Python/Test/FallImage/Cam/test.jpg", image_bgr)
                        final_image = np.concatenate((image_bgr, final_image), axis=1)

                        # compare stored pelvis coordinate
                        current_pelvis = adjusted_joints_coord[0][8]
                        length_pelvis_queue = len(list_pelvis_queue[index_cap])
                        if length_pelvis_queue != 0:
                            average_stored_pelvis_x = 0
                            average_stored_pelvis_y = 0
                            # b_temp_movement = false
                            for index_stored_pelvis in range(length_pelvis_queue):
                                average_stored_pelvis_x += list_pelvis_queue[index_cap][index_stored_pelvis][0]
                                average_stored_pelvis_y += list_pelvis_queue[index_cap][index_stored_pelvis][1]
                            average_stored_pelvis_x /= length_pelvis_queue
                            average_stored_pelvis_y /= length_pelvis_queue
                            if abs(current_pelvis[0] - average_stored_pelvis_x) < stasis_threshold_x and abs(current_pelvis[1] - average_stored_pelvis_y) < stasis_threshold_y:
                                list_b_person_movement_queue[index_cap].append(False)
                                # print("room ", str(index_cap), " static!")
                            else:
                                list_b_person_movement_queue[index_cap].append(True)
                                # print("room ", str(index_cap), " moving!")

                            # b_movement = False
                            # for index_stored_pelvis in range(length_pelvis_queue):
                            #     stored_frame_pelvis_x = list_pelvis_queue[index_cap][index_stored_pelvis][0]
                            #     stored_frame_pelvis_y = list_pelvis_queue[index_cap][index_stored_pelvis][1]
                            #     if abs(current_pelvis[0] - stored_frame_pelvis_x) > stasis_threshold_x and abs(current_pelvis[1] - stored_frame_pelvis_y) > stasis_threshold_y:
                            #         b_movement = True
                            #         break
                            # if b_movement:
                            #     list_b_person_movement_queue[index_cap].append(True)
                            # else:
                            #     list_b_person_movement_queue[index_cap].append(False)

                        else:
                            list_b_person_movement_queue[index_cap].append(True)

                        if length_pelvis_queue == 5:
                            list_pelvis_queue[index_cap].pop(0)
                        list_pelvis_queue[index_cap].append(current_pelvis)
                        if len(list_b_person_movement_queue[index_cap]) >= 8:
                            list_b_person_movement_queue[index_cap].pop(0)

                        # fall-down check:
                        fall_down_list = frame_pose.fall_down_check(adjusted_joints_coord, input_height)
                        if(len(fall_down_list) > 0):
                            list_fall_down_frame_counter[index_cap] += 1
                            # print("fall!")
                        else:
                            list_fall_down_frame_counter[index_cap] = 0

                        if list_b_person_in_room[index_cap] is False:
                            list_b_person_in_room[index_cap] = True
                            str_log = datetime.datetime.now().strftime(ISOTIMEFORMAT) + ": enter room " + str(index_cap) + " \n"
                            print(str_log)
                            with open('log.txt', 'a') as file_log_entrance_and_exit:
                                file_log_entrance_and_exit.write(str_log)

                        # temp mosaic 2022.3.17
                        for batch_index, batch_image_joints in enumerate(adjusted_joints_coord):
                            joint_nose = batch_image_joints[0]
                            joint_neck = batch_image_joints[1]
                            dist = math.sqrt(pow(joint_neck[0] - joint_nose[0], 2) + pow(joint_neck[1] - joint_nose[1], 2))
                            do_mosaic(final_image, int(joint_nose[0] - dist), int(joint_nose[1] - dist), int(2 * dist), int(2 * dist))

                    else:   # no person found in the frame
                        final_image = np.concatenate((image_bgr, img_black.copy()), axis=1)
                        # final_image = image_bgr

                        if list_b_person_in_room[index_cap] is True:
                            list_b_person_in_room[index_cap] = False
                            str_log = datetime.datetime.now().strftime(ISOTIMEFORMAT) + ": leave room " + str(index_cap) + " \n"
                            print(str_log)
                            with open('log.txt', 'a') as file_log_entrance_and_exit:
                                file_log_entrance_and_exit.write(str_log)

                        # clear the record queue
                        if(len(list_pelvis_queue[index_cap]) != 0):
                            list_pelvis_queue[index_cap].clear()
                        if(len(list_b_person_movement_queue[index_cap]) != 0):
                            print("clear person movement queue")
                            list_b_person_movement_queue[index_cap].clear()

                    # decide whether the person has fallen down or not
                    # print bounding box on image
                    for box_index, box in enumerate(square_box_list):
                        (x1, y1), (x2, y2) = box
                        box = [x1, y1, x2, y2]
                        if list_fall_down_frame_counter[index_cap] > fall_down_frame_threshold:
                            print(datetime.datetime.now().strftime(ISOTIMEFORMAT))
                            # send fall down alert to server
                            payload = {
                                "id": "testCameraId3",
                                "cameraImageUrl": "/home/pose/Workspace/Python/Test/FallImage/Cam/test.jpg",
                                "poseImageUrl": "/home/pose/Workspace/Python/Test/FallImage/Pose/test.jpg",
                                "type": 1,
                                "alarmTime": datetime.datetime.now().strftime(ISOTIMEFORMAT)
                            }
                            r = requests.post(url_http, headers=headers, data=json.dumps(payload))
                            print(r.status_code)
                            print(r.content)

                            color = (255, 0, 0) if (fall_down_list.count(box_index) != 0) else (255, 0, 0)
                            # color = (0, 0, 255) if (fall_down_list.count(box_index) != 0) else (255, 0, 0)
                        else:
                            color = (255, 0, 0)
                        plot_box(final_image, box, "xyxy", color)

                    # print moving or static on image
                    if list_b_person_in_room[index_cap] is False:
                        cv2.putText(final_image, "nobody", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 100, 255), 5)
                    else:
                        move_frame_count = 0
                        for index_list_b_move in range(len(list_b_person_movement_queue[index_cap])):
                            if list_b_person_movement_queue[index_cap][index_list_b_move] is True:
                                move_frame_count += 1

                        if move_frame_count >= len(list_b_person_movement_queue[index_cap]) * 0.6:
                            cv2.putText(final_image, "moving", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 100, 255), 5)
                        else:
                            cv2.putText(final_image, "static", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 100, 255), 5)

                    list_final_image.append(final_image)

        if len(list_final_image) > 0 and b_available_cap_exsist:
            for i in range(len(list_final_image)):
                if i == 0:
                    image_concatenate = list_final_image[0]
                else:
                    image_concatenate = np.concatenate((image_concatenate, list_final_image[i]), axis=0)
            # image_concatenate = cv2.resize(image_concatenate, (2000, 500 * len(list_final_image)))
            cv2.imshow("img", image_concatenate)
            # out.write(image_concatenate)
            cv2.waitKey(1)

        # reset the frame counter
        if counter % interval == 24:
            counter = 0
        counter += 1
    for i in range(len(list_cap)):
        cap_temp = list_cap[i]
        cap_temp.release()
    # out.release()
    # plt.show()
