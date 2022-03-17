from cgi import test
from curses.ascii import SP
from ntpath import join
from turtle import Turtle
from zipfile import ZipFile
from matplotlib import projections
from matplotlib.style import available
from pyrender import Camera
import torchvision
import cv2
import torch
import numpy as np

from torchvision.transforms import Normalize
import argparse
import json
from models import hmr, SMPL
from utils.imutils import crop
import config
import constants
import matplotlib.pyplot as plt    # draw result. by cococat 2021.12.28
from mpl_toolkits.mplot3d import Axes3D
import datetime
import frame_pose

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
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


def pixel2camera(K, uv):
    """
    K: 3x3
    uv: 2
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u, v = uv[0], uv[1]
    X = (u - cx) / fx
    Y = (v - cy) / fy
    Z = 1
    return np.array([X, Y, Z])


def camera2world(R, t, p):
    """
    R: 3x3
    t: 3,
    p: 3,
    """
    # matlab 先转置
    R = R.T
    # R_inv = np.linalg.inv(R)
    # return R.dot(p - t)
    return (p - t).dot(R)


def transFromImage2Camera(K_matrix, pt):
    ret = np.array([
        (pt[0] - K_matrix[0][2]) * pt[2] / K_matrix[0][0],
        (pt[1] - K_matrix[1][2]) * pt[2] / K_matrix[1][1],
        1.0
    ])
    return ret


def transFromCamera2World(R_matrix, C_matrix, pt):
    ret = np.array([
        # R_matrix[0][0] * pt[0] + R_matrix[1][0] * pt[1] + R_matrix[2][0] * pt[2] + C_matrix[0],
        # R_matrix[0][1] * pt[0] + R_matrix[1][1] * pt[1] + R_matrix[2][1] * pt[2] + C_matrix[1],
        # R_matrix[0][2] * pt[0] + R_matrix[1][2] * pt[1] + R_matrix[2][2] * pt[2] + C_matrix[2]
        R_matrix[0][0] * pt[0] + R_matrix[0][1] * pt[1] + R_matrix[0][2] * pt[2] + C_matrix[0],
        R_matrix[1][0] * pt[0] + R_matrix[1][1] * pt[1] + R_matrix[1][2] * pt[2] + C_matrix[1],
        R_matrix[2][0] * pt[0] + R_matrix[2][1] * pt[1] + R_matrix[2][2] * pt[2] + C_matrix[2]
    ])
    return ret


def transFromCamera2World_matlab(R_matrix_transpose, T_matrix, pt):
    return np.array([
        (pt[0] - T_matrix[0]) * R_matrix_transpose[0][0] + (pt[1] - T_matrix[1]) * R_matrix_transpose[1][0] + (pt[2] - T_matrix[2]) * R_matrix_transpose[2][0],
        (pt[0] - T_matrix[0]) * R_matrix_transpose[0][1] + (pt[1] - T_matrix[1]) * R_matrix_transpose[1][1] + (pt[2] - T_matrix[2]) * R_matrix_transpose[2][1],
        (pt[0] - T_matrix[0]) * R_matrix_transpose[0][2] + (pt[1] - T_matrix[1]) * R_matrix_transpose[1][2] + (pt[2] - T_matrix[2]) * R_matrix_transpose[2][2]
    ])


def transFromImage2World_matlab(K_martix, R_matrix, T_matrix, pt):
    # K_inv = np.linalg.inv(K_martix)
    R_T_matrix = np.vstack((R_matrix, T_matrix))
    Camera_matrix = np.matmul(R_T_matrix, K_martix)
    Cam_inv_matrix = np.linalg.pinv(Camera_matrix)
    world_coordinate_homo = np.matmul(pt, Cam_inv_matrix)

    world_coordinate = np.array([
        world_coordinate_homo[0] / world_coordinate_homo[3],
        world_coordinate_homo[1] / world_coordinate_homo[3],
        world_coordinate_homo[2] / world_coordinate_homo[3]
        ])
    return world_coordinate


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
    cap = cv2.VideoCapture("http://192.168.1.201:8080/proxy/0.flv")    

    list_cap = [cap]

    input_width, input_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame extraction
    interval = 3  # 3
    counter = 0

    # save result as file
    input_FPS = cap.get((cv2.CAP_PROP_FPS))
    extracted_FPS = input_FPS / interval
    fourCC = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter("/home/pose/Videos/debug.mp4", fourCC, extracted_FPS, (2000, 500))

    # read calibration result from file
    K_matrix = readMatrixFromFile("/home/pose/Workspace/Python/Test/K.txt")
    R_matrix = readMatrixFromFile("/home/pose/Workspace/Python/Test/R.txt")
    T_matrix = readMatrixFromFile("/home/pose/Workspace/Python/Test/T.txt")

    # R_matrix_inv = np.linalg.inv(R_matrix)
    R_matrix_transpose = np.transpose(R_matrix)
    # R_matrix, _  = cv2.Rodrigues(R_matrix)
    # R_matrix = np.transpose(R_matrix)
    # C_matrix = np.matmul(-np.linalg.inv(R_matrix), T_matrix)
    # C_matrix = -np.matmul(R_matrix, T_matrix)
    # C_matrix = -np.matmul(np.transpose(R_matrix), T_matrix)

    fall_down_frame_counter = 0
    fall_down_frame_threshold = extracted_FPS / 2  # falled down for 0.5 second
    fall_down_list = []
    list_fall_down_frame_counter = [0 for n in range(len(list_cap))]

    # initialize 3D showcase by cococat 2022.3.2
    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_xlim(2, 3)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(15, 20)
    azim = -60
    elev = 30

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
                        cv2.imwrite("/home/pose/Pictures/skeleton.jpg", final_image)
                        cv2.imwrite("/home/pose/Pictures/origin.jpg", image_bgr)
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

                        # project from image coordinate to world coordinate
                        SPIN_camera_joints = SPIN_camera_joints.cpu().numpy()
                        SPIN_camera_joints_XZY = []
                        for iter in range(SPIN_camera_joints.shape[0]):
                            temp_joints = SPIN_camera_joints[iter]
                            SPIN_camera_joints_XZY.append(temp_joints[:, [0, 2, 1]])

                        # frame_pose.save_joints_obj("/home/pose/Workspace/Python/Test/Joint_SPIN.obj", SPIN_camera_joints_XZY[0])
                        list_all_joint_world_coord = []
                        for k in range(adjusted_joints_coord.shape[0]):
                            list_current_person_joint = []
                            for j in range(adjusted_joints_coord[0].shape[0]):
                                temp_img_joint_coord = np.array([adjusted_joints_coord[k][j][0], adjusted_joints_coord[k][j][1], 1])
                                # temp_cam_joint_coord = transFromImage2Camera(K_matrix, temp_img_joint_coord)
                                # # temp_world_joint_coord = transFromCamera2World(R_matrix, C_matrix, temp_cam_joint_coord)
                                # temp_world_joint_coord = transFromCamera2World_matlab(R_matrix_transpose, T_matrix, temp_cam_joint_coord)
                                # # temp_world_joint_coord = transFromImage2World_matlab(K_matrix, R_matrix, T_matrix, temp_img_joint_coord)


                                # temp_cam_joint_coord = pixel2camera(K_matrix, temp_img_joint_coord)
                                # temp_world_joint_coord = camera2world(R_matrix, T_matrix, temp_cam_joint_coord)


                                # SPIN_joint_XZY = np.array([SPIN_camera_joints[k][j][0], SPIN_camera_joints[k][j][2], SPIN_camera_joints[k][j][1]])
                                # temp_world_joint_coord = camera2world(R_matrix, np.array([0,0,0]), SPIN_joint_XZY)
                                temp_world_joint_coord = camera2world(R_matrix, np.array([0,0,0]), SPIN_camera_joints[k][j])

                                # temp_world_joint_coord = camera2world(R_matrix, np.array([0, 0, 0]), SPIN_camera_joints_XZY[k][j])

                                list_current_person_joint.append(temp_world_joint_coord)
                            list_all_joint_world_coord.append(list_current_person_joint)
                        
                        list_all_joint_world_coord = np.array(list_all_joint_world_coord)
                        # frame_pose.save_joints_obj("/home/pose/Workspace/Python/Test/Joint_Rotated.obj", list_all_joint_world_coord[0])

                        # print(list_all_joint_world_coord[0, 8])

                        # # real-time draw joints in world coordinate
                        # plt.clf()
                        # fig = plt.gcf()
                        # ax = fig.gca(projection='3d')
                        # ax.view_init(elev, azim)

                        # ax.scatter(list_all_joint_world_coord[0, :, 0], list_all_joint_world_coord[0, :, 1], list_all_joint_world_coord[0, :, 2])
                        # # test_joints_np = SPIN_camera_joints.cpu().numpy()
                        # # ax.scatter(test_joints_np[0, :, 0], test_joints_np[0, :, 1], test_joints_np[0, :, 2])

                        # ax.set_xlabel('X')
                        # ax.set_ylabel('Y')
                        # ax.set_zlabel('Z')
                        # # plt.axis("equal")
                        # # plt.xticks(np.linspace(0, 0.1, 2))
                        # # plt.yticks(np.linspace(0, 0.1, 2))
                        # # plt.zticks(np.linspace(0, 0.1, 2))
                        # plt.pause(0.001)

                        # elev, azim = ax.elev, ax.azim

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
                    # print bounding box on image
                    for box_index, box in enumerate(square_box_list):
                        (x1, y1), (x2, y2) = box
                        box = [x1, y1, x2, y2]
                        if list_fall_down_frame_counter[index_cap] > fall_down_frame_threshold:
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
            image_concatenate = cv2.resize(image_concatenate, (2000, 500 * len(list_final_image)))
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
