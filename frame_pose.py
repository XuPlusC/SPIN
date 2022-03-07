from numpy.core.defchararray import join
from torch import tensor
from torch._C import _jit_pass_onnx_assign_output_shape
import torchvision
import cv2
import torch
import numpy as np
from torchvision.models.resnet import wide_resnet101_2

from torchvision.transforms import Normalize
import argparse
import json
from models import hmr, SMPL
from utils.imutils import crop
import config
import constants
import matplotlib.pyplot as plt    # draw result. by cococat 2021.12.28
import math


def save_joints_obj(filename, joint_3d):
    with open(filename, "w") as fw:
        for i in range(joint_3d.shape[0]):
            x = joint_3d[i][0]
            y = joint_3d[i][1]
            z = joint_3d[i][2]
            # r, g, b = float(i + 1) / 49, float(i + 1) / 49, float(i + 1) / 49
            r, g, b = 0.8, 0.8, 0.8
            fw.write(f"v {x} {y} {z} {r} {g} {b}\n")


# def process_image(img_file, bbox_list, openpose_file, input_res=224):
def process_image(img_input_rgb, bbox_list, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = img_input_rgb.copy()  # PyTorch does not support negative stride at the moment
    # img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    # img_origincv = cv2.imread(img_file)

    normalize_imgs = []
    bbox_centers = []
    bbox_scales = []

    for box in bbox_list:
        # ul_corner = box[0]  # upper left
        # br_corner = box[1]  # bottom right
        box_height = box[1][1] - box[0][1]
        box_width = box[1][0] - box[0][0]
        box_center = np.array([box[0][0] + box_width / 2, box[0][1] + box_height / 2])
        box_scale = box_width / 200.0
        temp_crop_img = crop(img, box_center, box_scale, (input_res, input_res))
        # cv2.imwrite('batch.png', temp_crop_img[:,:,::-1])
        temp_crop_img = temp_crop_img.astype(np.float32) / 255.
        temp_crop_img = torch.from_numpy(temp_crop_img).permute(2, 0, 1)
        norm_img = normalize_img(temp_crop_img.clone())[None]

        normalize_imgs.append(norm_img)
        bbox_centers.append(box_center)
        bbox_scales.append(box_scale)

    # img_origincv = crop(img_origincv, center, scale, (input_res, input_res))
    return torch.cat(normalize_imgs, dim=0), bbox_centers, bbox_scales


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    # TODO: batchsize!
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def adjust_joints_2D_coordinate(img_joints, bbox_centers, bbox_scales):
    adjusted_joints = []

    for batch_index, batch_image_joints in enumerate(img_joints):
        current_image_joints = []
        for joint_index, joint_coord in enumerate(batch_image_joints):
            if joint_index >= 19:
                break
            new_joint = np.array([joint_coord[0] * bbox_scales[batch_index] + bbox_centers[batch_index][0], 
                            joint_coord[1] * bbox_scales[batch_index] + bbox_centers[batch_index][1]])
            current_image_joints.append(new_joint)
        adjusted_joints.append(current_image_joints)

    return np.array(adjusted_joints)


def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            # Format: each point should contain (y, x, confidence)
            Format: each point should contain (x, y)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    # bias = constants.IMG_RES // 2
    bias = 0

    for batch_index, batch_points in enumerate(points):
        for i, joint in enumerate(skeleton):
            pt1, pt2 = batch_points[joint]
            image = cv2.line(
                # image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                image, (int(pt1[0]) + bias, int(pt1[1]) + bias), (int(pt2[0]) + bias, int(pt2[1]) + bias),
                tuple(colors[person_index % len(colors)]), 2
            )

    return image


def draw_points(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            # Format: each point should contain (y, x, confidence)
            Format: each point should contain (x, y)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid points

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))

    # bias = constants.IMG_RES // 2
    bias = 0

    for batch_index, batch_points in enumerate(points):
        for i, pt in enumerate(batch_points):
                image = cv2.circle(image, (int(pt[0]) + bias, int(pt[1]) + bias), circle_size, tuple(colors[i % len(colors)]), -1)

    return image


'''
    joint index definition of joints in SMPL.joints
    0 nose
    1 neck
    2 right shoulder
    3 right elbow
    4 right wrist
    5 left shoulder
    6 left elbow
    7 left wrist
    8 pelvis
    9 right hip
    10 right knee
    11 right ankle
    12 left hip
    13 left knee
    14 left ankle
    15 right eye
    16 left eye
    17 right ear
    18 left ear
    19 20 21 left foot
    22 23 24 right foot
    by cococat 2021.12.29
'''
skeleton = [
    [0, 1], [0, 15], [0, 16], [15, 17], [16, 18],   # head
    [1, 2], [2, 3], [3, 4],         # right arm
    [1, 5], [5, 6], [6, 7],         # left arm
    [1, 8],                         # spine(neck -> pelvis)
    [8, 9], [9, 10], [10, 11],      # right leg
    [8, 12], [12, 13], [13, 14],    # left leg
]


def draw_points_and_skeleton(image, points, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0

    Returns:
        A new image with overlaid joints

    """
    image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                          palette_samples=skeleton_palette_samples, person_index=person_index)
    image = draw_points(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples)
    return image


def joint_k_check(joint1, joint2):
    k = 0
    if(joint1.size == 2):
        delta_x = abs(joint1[0] - joint2[0])
        if(delta_x == 0):
            k = 100
        else:
            k = abs(joint1[1] - joint2[1]) / delta_x
    if(joint1.size == 3):
        delta_x = joint1[0] - joint2[0]
        delta_y = joint1[1] - joint2[1]
        delta_z = joint1[2] - joint2[2]
        # delta_x_y_sqrt = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        delta_x_z_sqrt = math.sqrt(delta_x * delta_x + delta_z * delta_z)
        if (delta_x_z_sqrt == 0):
            k = 100
        else:
            k = abs(delta_y) / delta_x_z_sqrt
    # print("k: ", k)
    return 1 if k <= 1 else 0


def fall_down_check(img_joints, video_height):
    fall_down_list = []
    falldown_threshold = video_height - video_height / 3

    for batch_index, batch_image_joints in enumerate(img_joints):
        # current_image_joints = []
        # for joint_index, joint_coord in enumerate(batch_image_joints):
        #     if joint_index >= 19:
        #         break
        #     new_joint = np.array([joint_coord[0] * bbox_scales[batch_index] + bbox_centers[batch_index][0], 
        #                     joint_coord[1] * bbox_scales[batch_index] + bbox_centers[batch_index][1]])
        #     current_image_joints.append(new_joint)
        # fall_down_list.append(current_image_joints)
        joint_2D_coord_too_low_check = 0
        for i in range(15):
            if(batch_image_joints[i][1] > falldown_threshold):
                joint_2D_coord_too_low_check += 1
        # print("joint 2d coord counter : ", joint_2D_coord_too_low_check)
        if joint_2D_coord_too_low_check > 10:
            fall_down_list.append(batch_index)
            continue

        neck = batch_image_joints[1]
        pelvis = batch_image_joints[8]
        r_hip = batch_image_joints[9]
        r_knee = batch_image_joints[10]
        r_ankle = batch_image_joints[11]
        l_hip = batch_image_joints[12]
        l_knee = batch_image_joints[13]
        l_ankle = batch_image_joints[14]
        joint_check = 0
        joint_check += joint_k_check(neck, pelvis)      # k_spine
        joint_check += joint_k_check(r_hip, r_knee)     # k_r_thigh
        joint_check += joint_k_check(r_knee, r_ankle)   # k_r_crus
        joint_check += joint_k_check(l_hip, l_knee)     # k_l_thigh
        joint_check += joint_k_check(l_knee, l_ankle)   # k_l_curs
        joint_check += joint_k_check(neck, r_ankle)
        joint_check += joint_k_check(neck, l_ankle)

        if(joint_check >= 4):
            fall_down_list.append(batch_index)

    return fall_down_list


def fall_down_check_3D(joints):
    fall_down_list = []
    aroundx = cv2.Rodrigues(np.array([np.radians(30.), 0, 0]))[0]
    # center = pred_vertices.mean(axis=0)
    # rot_vertices = np.dot((pred_vertices - center), aroundy) + center

    for batch_index, batch_joints in enumerate(joints):
        center = batch_joints.mean(axis=0)
        save_joints_obj("/home/pose/Workspace/Python/Test/origin_joints.obj", batch_joints)
        rot_joints = np.dot((batch_joints - center), aroundx) + center
        save_joints_obj("/home/pose/Workspace/Python/Test/rot_joints.obj", rot_joints)
        neck = rot_joints[1]
        pelvis = rot_joints[8]
        r_hip = rot_joints[9]
        r_knee = rot_joints[10]
        r_ankle = rot_joints[11]
        l_hip = rot_joints[12]
        l_knee = rot_joints[13]
        l_ankle = rot_joints[14]
        joint_check = 0
        joint_check += joint_k_check(neck, pelvis)      # k_spine
        joint_check += joint_k_check(r_hip, r_knee)     # k_r_thigh
        joint_check += joint_k_check(r_knee, r_ankle)   # k_r_crus
        joint_check += joint_k_check(l_hip, l_knee)     # k_l_thigh
        joint_check += joint_k_check(l_knee, l_ankle)   # k_l_curs
        # print("joint check: ", joint_check)
        if(joint_check >= 3):
            fall_down_list.append(batch_index)

    return fall_down_list


if __name__ == '__main__':
    pts1 = np.float32([[0, 0], [0, 100], [100, 0], [100, 100]])
    pts2 = np.float32([[10, 0], [0, 100], [90, 0], [100, 100]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)
    img = cv2.imread("XJXTest/batch0.png")
    res = cv2.warpPerspective(img, M, (200, 200))
    cv2.imshow("input", img)
    cv2.imshow("res", res)
    cv2.waitKey(0)