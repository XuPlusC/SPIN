import cv2
import numpy as np
import glob

# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 棋盘格模板规格
w = 11
h = 8
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

# images = glob.glob('C:/Users/ZQQ/Desktop/advanced/study/computervision/images/ch05/*.jpg')
cap = cv2.VideoCapture("rtsp://cococat:f51e7e7951598caf878fb26dc67d58cf@192.168.162.95/67")
if (cap.isOpened()):
    ret, img = cap.read()
    img_undistort = img
    if(ret is True):
        # img = cv2.imread(fname)
        # img = cv2.imread("/home/pose/Workspace/Python/Test/checkerboard.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # 如果找到足够点对，将其存储起来
        if ret is True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            cv2.imshow('findCorners', img)
            cv2.waitKey(1)
            # i += 1
            cv2.imwrite('corner.jpg', img)
            # 标定
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            print("ret:", ret)
            print("mtx:\n", mtx)        # 内参数矩阵
            print("dist:\n", dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
            print("rvecs:\n", rvecs)    # 旋转向量  # 外参数
            print("tvecs:\n", tvecs)   # 平移向量  # 外参数

            print("-----------------------------------------------------")
            # 去畸变
            # img_undistort = cv2.imread(images[2])
            h, w = img_undistort.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h)) # 自由比例参数
            dst = cv2.undistort(img_undistort, mtx, dist, None, newcameramtx)
            print("newcameramtx:\n", newcameramtx)
            # 根据前面ROI区域裁剪图片
            # x,y,w,h = roi
            # dst = dst[y:y+h, x:x+w]
            cv2.imshow('findCorners', dst)
            cv2.waitKey(1)
            cv2.imwrite('calibresult.png', dst)

            # 反投影误差
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                total_error += error
            print("total error: ", total_error/len(objpoints))

        else:
            print("no checkerboard found!")
