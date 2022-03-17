import numpy as np
import cv2
import math


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
    # R = np.linalg.inv(R)
    return (p - t).dot(R)


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I_ = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I_ - shouldBeIdentity)
    return n < 1e-6


# R-P-Y
def rotationMatrixToEulerAngles(R):

    print(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # return np.array([x, y, z])
    return np.array([x * 180 / math.pi, y * 180 / math.pi, z * math.pi / 180])



def saveCalibrationImage():
    cap = cv2.VideoCapture("rtsp://cococat:f51e7e7951598caf878fb26dc67d58cf@192.168.162.95/67")
    savedCounter = 0
    # Frame extraction
    interval = 15  # 3
    frameCounter = 0
    while (cap.isOpened()):
        ret, image_bgr = cap.read()
        if(ret is not True):
            continue

        if interval == 0 or frameCounter % interval == 0:
            cv2.imwrite("/home/pose/Workspace/Python/Test/calibrationImages/" + str(savedCounter) + ".jpg", image_bgr)
            cv2.imshow("img", image_bgr)
            cv2.waitKey(1)
            savedCounter += 1
        if frameCounter % interval == 0:
            frameCounter = 0
        frameCounter += 1


if __name__ == "__main__":
    # R_matrix = np.loadtxt("/home/pose/Workspace/Python/Test/R.txt")
    newR = np.array([
        [-0.6896, 0.4736, -0.5479],
        [-0.7242, -0.4539, 0.5191],
        [-0.0028, 0.7547, 0.6560]
    ])

    # T_vector = np.loadtxt("/home/pose/Workspace/Python/Test/T.txt")
    # world_coord = camera2world(R_matrix, T_vector, T_vector)
    # print("world Coord:", world_coord)
    # print("R: ", R_matrix)
    # print("T: ", T_vector)
    # print("R inv", np.linalg.inv(R_matrix))
    # print("R trans", R_matrix.T)
    
    euler_angle = rotationMatrixToEulerAngles(newR.T)
    print(euler_angle)
