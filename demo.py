"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

from numpy.core.arrayprint import set_string_function
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants

import matplotlib.pyplot as plt    # draw result. by cococat 2021.12.28
import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

def save_smpl_obj(filename, vertices, faces=None, colors=None):
    """保存smpl obj文件"""

    if colors is not None:
        assert colors.shape == vertices.shape

    with open(filename, "w") as fw:
        if colors is None:
            for i in range(vertices.shape[0]):
                x, y, z = vertices[i]
                r, g, b = 0.5, 0.5, 0.5
                fw.write(f"v {x} {y} {z} {r} {g} {b}\n")
        else:
            for i in range(vertices.shape[0]):
                x, y, z = vertices[i]
                r, g, b = colors[i]
                fw.write(f"v {x} {y} {z} {r} {g} {b}\n")

        if faces is not None:
            for i in range(faces.shape[0]):
                a = faces[i][0]
                b = faces[i][1]
                c = faces[i][2]
                fw.write(f"f {a+1} {b+1} {c+1}\n")


def save_smpl_obj_jointOnly(filename, joint_3d):
    """保存smpl obj文件, 仅保存骨架点"""
    # assert joint_3d.shape[0] >= 23

    with open(filename, "w") as fw:
        for i in range(joint_3d.shape[0]):
            # if(i >= 23):
            #     break
            x = joint_3d[i][0]
            y = joint_3d[i][1]
            z = joint_3d[i][2]
            r, g, b = float(i + 1) / 49, float(i + 1) / 49, float(i + 1) / 49
            # r, g, b = 0.8, 0.8, 0.8
            fw.write(f"v {x} {y} {z} {r} {g} {b}\n")


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

    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        image = cv2.line(
            image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
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

    for i, pt in enumerate(points):
        # if pt[2] > confidence_threshold:
        image = cv2.circle(image, (int(pt[0]), int(pt[1])), circle_size, tuple(colors[i % len(colors)]), -1)

    return image


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


def convertFromCam2Image(points_cam, u, v):
    pt_image = np.empty((points_cam.shape[0], 2), dtype=np.float32)
    # for pt_cam in points_cam:
    iter = 0
    while iter < points_cam.shape[0]:
        pt_image[iter, 0] = constants.FOCAL_LENGTH * points_cam[iter, 0] / points_cam[iter, 2] + u
        pt_image[iter, 1] = constants.FOCAL_LENGTH * points_cam[iter, 1] / points_cam[iter, 2] + v
        iter += 1
    return pt_image


def convertFromWorld2Cam(points_world, camera_T):
    pt_cam = np.empty((points_world.shape[0], 3), dtype=np.float32)
    # for pt_cam in points_world:
    iter = 0
    while iter < points_world.shape[0]:
        pt_cam[iter, 0] = points_world[iter, 0] - camera_T[0]
        pt_cam[iter, 1] = points_world[iter, 1] + camera_T[1]
        pt_cam[iter, 2] = points_world[iter, 2] + camera_T[2]
        iter += 1
    return pt_cam


if __name__ == '__main__':
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    # Preprocess input image and generate predictions
    img, norm_img = process_image(args.img, args.bbox, args.openpose, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))   # 人体参数
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        
    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()  # 顶点 保存成obj 
    img = img.permute(1,2,0).cpu().numpy()

    save_smpl_obj("/home/pose/Workspace/Python/Test/test.obj", pred_vertices)

    test_joints = pred_output.joints
    test_joints = test_joints[0].cpu().numpy()
    save_smpl_obj_jointOnly("/home/pose/Workspace/Python/Test/joint49.obj", test_joints)

    result_img = cv2.imread(args.img)
    cam_joints = convertFromWorld2Cam(test_joints[:19], camera_translation)
    img_joints = convertFromCam2Image(cam_joints, result_img.shape[1] / 2, result_img.shape[0] / 2)
    cv2.namedWindow("image")
    # cv2.imshow('image', result_img)
    # cv2.waitKey(0)
    result_img = draw_points_and_skeleton(result_img, img_joints, skeleton)
    cv2.imshow('image', result_img)
    cv2.waitKey(0)

    
    # # Render parametric shape
    # img_shape = renderer(pred_vertices, camera_translation, img)
    
    # # # test: render joints
    # # img_joints = renderer(test_joints, camera_translation, img)

    # # Render side views
    # aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    # center = pred_vertices.mean(axis=0)
    # rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    
    # # Render non-parametric shape
    # img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

    # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

    # # Save reconstructions
    # cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:,:,::-1])
    # # cv2.imwrite(outfile + '_shape_joints.png', 255 * img_joints[:,:,::-1])
    # cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
