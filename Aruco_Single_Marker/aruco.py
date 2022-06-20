#encoding:utf-8
import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import serial

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

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

    return np.array([x, y, z])

#--- Define Tag
id_to_find  = 10
marker_size  = 5 #- [cm]

# 找棋盘格角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值

#棋盘格模板规格
w = 9   # 10 - 1
h = 6   # 7  - 1

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*18.1  # 18.1 mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('C\Users\CYH\Desktop\aruco\pictures*.jpg')  #   拍摄的十几张棋盘图片所在目录

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)

    # 如果找到足够点对，将其存储起来
    if ret == True:

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 810, 405)
        cv2.imshow('findCorners',img)
        cv2.waitKey(1)

cv2.destroyAllWindows()

#%% 标定
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#print("ret:",ret  )
print("mtx:\n",mtx)      # 内参数矩阵
print("dist:\n",dist   )   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
#print("rvecs:\n",rvecs)   # 旋转向量  # 外参数
#print("tvecs:\n",tvecs  )  # 平移向量  # 外参数

#--- Get the camera calibration path
#calib_path = ""

camera_matrix = mtx
camera_distortion = dist

#--- 180 deg rotation matrix around the x axis
R_flip      = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

#--- Define the aruco dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()

#--- Capture the videocamera (this may also be a video or a picture)
cap = cv2.VideoCapture(0)
#-- Set the camera size as the one it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

dev=serial.Serial('/dev/ttyUSB0',115200,timeout=0.5)
dev.readlines()

while True:
    #-- Read the camera frame
    ret, frame = cap.read()
    ax = 0
    ay = 0
    az = 0
    dx = 0
    dy = 0
    dz = 0
    #-- Convert in gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image = gray, dictionary = aruco_dict,
                    parameters = parameters,
                    cameraMatrix = camera_matrix,
                    distCoeff = camera_distortion)

    if ids != None and ids[0] == id_to_find:
        #-- ret = [rvec, tvec, ?]
        #-- array of rotation and position of each marker in camera frame
        #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
        #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        #-- Unpack the output, get only the first
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        #-- Draw the detected marker and put a reference frame over it
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

        #-- Print the tag position in camera frame
        str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (0, 100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        ax = tvec[0]  #-- x轴传递参数
        ay = tvec[1]  #-- y轴传递参数
        az = tvec[2]  #-- z轴传递参数
        #-- Obtain the rotation matrix tag->camera
        R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
        R_tc    = R_ct.T

        #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)

        #-- Print the marker's attitude respect to camera frame
        str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
                            math.degrees(yaw_marker))
        cv2.putText(frame, str_attitude, (0, 150), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        dx = math.degrees(roll_marker)  #-- 翻滚角传递参数
        dy = math.degrees(pitch_marker) #-- 俯仰角传递参数
        dz = math.degrees(yaw_marker)   #-- 偏航角传递参数
    try:
        #--- Display the frame
        cv2.imshow('frame', frame)
    except:
        None

	#-- 将数据通过串口传递至上位机
    dev.write("ax=".encode())
    if ax >= 0:
        dev.write("+".encode())
    if ax < 0:
        ax = -ax
        dev.write("-".encode())
    dev.write("{:.2f}\r\n".format(ax).encode())

    dev.write("ay=".encode())
    if ay >= 0:
        dev.write("+".encode())
    if ay < 0:
        ay = -ay
        dev.write("-".encode())
    dev.write("{:.2f}\r\n".format(ay).encode())

    dev.write("az=".encode())
    if az >= 0:
        dev.write("+".encode())
    if az < 0:
        az = -az
        dev.write("-".encode())
    dev.write("{:.2f}\r\n".format(az).encode())

    dev.write("\r\n".encode())

    dev.write("dx=".encode())
    if dx >= 0:
        dev.write("+".encode())
    if dx < 0:
        dx = -dx
        dev.write("-".encode())
    dev.write("{:.2f}\r\n".format(dx).encode())

    dev.write("dy=".encode())
    if dy >= 0:
        dev.write("+".encode())
    if dy < 0:
        dy = -dy
        dev.write("-".encode())
    dev.write("{:.2f}\r\n".format(dy).encode())

    dev.write("dz=".encode())
    if dz >= 0:
        dev.write("+".encode())
    if dz < 0:
        dz = -dz
        dev.write("-".encode())
    dev.write("{:.2f}\r\n".format(dz).encode())

    dev.write("\r\n\r\n".encode())

    #--- use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


























"""
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 找棋盘格角点

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
#棋盘格模板规格
w = 9   # 10 - 1
h = 6   # 7  - 1
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*18.1  # 18.1 mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('/home/pi/LJQ/Aruco/picture/*.jpg')  #   拍摄的十几张棋盘图片所在目录

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 810, 405)
        cv2.imshow('findCorners',img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
#%% 标定
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


#print("ret:",ret  )
print("mtx:\n",mtx)      # 内参数矩阵
print("dist:\n",dist   )   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
#print("rvecs:\n",rvecs)   # 旋转向量  # 外参数
#print("tvecs:\n",tvecs  )  # 平移向量  # 外参数
"""

