#encoding:utf-8
"""
该演示针对不同的场景计算多项内容。
如果在 PI 上运行，请务必 sudo modprobe bcm2835-v4l2
以下是定义的参考框架：
标签：
                y
                |
                |
                |标签中心
                O--> x
相机：
                X--------> X
                | 框架中心
                |
                |
                维
F1：围绕 x 轴翻转（180 度）标签框
F2：围绕 x 轴翻转（180 度）相机框架
通用框架2相对于框架1的姿态可以通过计算euler(R_21.T)获得
我们将获得以下数量：
    > 从 aruco 库我们获得 tvec 和 Rct，标签在相机帧中的位置和标签的姿态
    > 相机在标签轴上的位置：-R_ct.T*tvec
    > 相机的变换，关于 f1（标签翻转帧）：R_cf1 = R_ct*R_tf1 = R_cf*R_f
    > 标签的变换，关于 f2（相机翻转的帧）：R_tf2 = Rtc*R_cf2 = R_tc*R_f
    > R_tf1 = R_cf2 对称 = R_f
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import glob
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# # 伽马变换
def gamma_adjust(im, gamma=1.0,is_hsv=False):
    """伽马矫正"""

    if is_hsv:
        img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)

        v_gamma =( np.power(v.astype(np.float32) / 255, 1 / gamma) * 255).astype(np.uint8)
        img_hsv_gamma=cv2.merge((h,s,v_gamma))
        img_gamma=cv2.cvtColor(img_hsv_gamma,cv2.COLOR_HSV2BGR)
    else:
        img_gamma=( np.power(im.astype(np.float32) / 255, 1 / gamma) * 255).astype(np.uint8)
    return img_gamma

def auto_gama_correction(image):
    
    #将BGR图像转换为HSV图像，并使用v通道获取图像的正确值
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv/255.0
    v = hsv[:, :, 2]
    
    #使用v通道计算图像的平均值和分布
    mu = v.mean()
    sigma = v.std()
    print("Mean : ", mu)
    #print("Stdev : ", sigma)
    
    #计算D值以确定低对比度或高对比度图像
    #其中D=4*sigma
    D = ((mu + (2*sigma)) - (mu - (2*sigma)))
    tres = 1 / 2
    Flag = ""
    if D < tres or D == tres:
      #print("Low contrast Image")
      Flag = "Low"
    else:
      #print("High contrast Image")
      Flag = "High"
      
    
    #确定图像是亮还是暗
    if mu >= 0.50:
      print("Bright")
    else:
      print("Dark")
    
    
    #计算伽马值
    if Flag == "High":
      gamma = np.exp((1 - (mu + sigma)) / 2)
    else:
      gamma = -np.log(sigma)    
      
    gamma = gamma.astype(np.float32)

    #print("gamma value : ", gamma)
    
    img_gamma = gamma_adjust(image, gamma,is_hsv = True)
    return img_gamma
    
    """
    #计算C的值
    heaviside_x = (0.50 - mu)

    if heaviside_x <= 0:
      heaviside = 0
    else:
      heaviside = 1
    
    #print("heaviside : ", heaviside)
    power_val = np.power(image/255 , gamma)
    power_val = power_val.astype(np.float32)

    k = power_val + ((1- power_val) * np.power(mu, gamma))
    k = k.astype(np.float32)
    c = 1 / (1 + (heaviside * (k -1)))
    c = c.astype(np.float32)
    image_out = c * (np.power(image/255, gamma))
    image_out = image_out.astype(np.float32)
    image_out = np.round(image_out * 255.0)
    #cv2.imwrite(output_path, image_out)
    return image_out
    """
    

#--- Define Tag
id_to_find  = 10
marker_size  = 6.2 #- [cm]

# 找棋盘格角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值

#棋盘格模板规格
w = 9   # 10 - 1
h = 6   # 7  - 1

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*30  # 18.1 mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('F:/Github/Aruco/Aruco_Single_Marker_Final/Aruco_Single_Marker/picture_calibration_color/*.jpg')  #   拍摄的十几张棋盘图片所在目录

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
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
parameters  = aruco.DetectorParameters_create()

#--- Capture the videocamera (this may also be a video or a picture)
cap = cv2.VideoCapture(1)

#-- Set the camera size as the one it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

while True:

    #-- Read the camera frame
    ret, frame = cap.read()
    #frame_gamma = auto_gama_correction(frame)
    #frame_gamma = gamma_adjust(frame,gamma = 2.2,is_hsv = True)
    frame_gamma = gamma_adjust(frame,gamma = 1 / 2.2,is_hsv = True)

    #-- Convert in gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red
    gray_frame_gamma = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red
    #gray_img_gamma = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

    #-- Find all the aruco markers in the image
    corners_frame, ids_frame, rejected_frame = aruco.detectMarkers(image = gray_frame, dictionary = aruco_dict,
                    parameters = parameters,
                    cameraMatrix = camera_matrix,
                    distCoeff = camera_distortion)
    
    corners_frame_gamma, ids_frame_gamma, rejected_frame_gamma = aruco.detectMarkers(image = gray_frame_gamma, dictionary = aruco_dict,
                    parameters = parameters,
                    cameraMatrix = camera_matrix,
                    distCoeff = camera_distortion)
    
    try:            
        if (ids_frame != None and ids_frame[0] == id_to_find):
          #-- ret = [rvec, tvec, ?]
          #-- array of rotation and position of each marker in camera frame
          #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
          #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
          ret_frame = aruco.estimatePoseSingleMarkers(corners_frame, marker_size, camera_matrix, camera_distortion)
          #ret_img_gamma = aruco.estimatePoseSingleMarkers(corners_img_gamma, marker_size, camera_matrix, camera_distortion)

          #-- Unpack the output, get only the first
          rvec_frame, tvec_frame = ret_frame[0][0,0,:], ret_frame[1][0,0,:]

          #-- Draw the detected marker and put a reference frame over it
          aruco.drawDetectedMarkers(frame, corners_frame)
          
          aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec_frame, tvec_frame, 10)

          #-- Print the tag position in camera frame
          str_position_frame = "MARKER Position x=%4.1f  y=%4.1f  z=%4.1f"%(tvec_frame[0], tvec_frame[1], tvec_frame[2])
          
          cv2.putText(frame, str_position_frame, (0, 100), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
          
          #-- Obtain the rotation matrix tag->camera
          R_ct_frame    = np.matrix(cv2.Rodrigues(rvec_frame)[0])
          
          R_tc_frame    = R_ct_frame.T

          #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
          roll_marker_frame, pitch_marker_frame, yaw_marker_frame = rotationMatrixToEulerAngles(R_flip*R_tc_frame)

          #-- Print the marker's attitude respect to camera frame
          str_attitude_frame = "MARKER Attitude r=%4.1f  p=%4.1f  y=%4.1f"%(math.degrees(roll_marker_frame),math.degrees(pitch_marker_frame),
                              90-math.degrees(yaw_marker_frame))
          """
          str_attitude_img_gamma = "MARKER Attitude r=%4.1f  p=%4.1f  y=%4.1f"%(math.degrees(roll_marker_img_gamma),math.degrees(pitch_marker_img_gamma),
                              90-math.degrees(yaw_marker_img_gamma))
          """
          
          cv2.putText(frame, str_attitude_frame, (0, 150), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            
        try:
            #--- Display the frame
            cv2.imshow('frame', frame)

        except:
            None
    
    except:
        None

    try:            
        if (ids_frame_gamma != None and ids_frame_gamma[0] == id_to_find):
          #-- ret = [rvec, tvec, ?]
          #-- array of rotation and position of each marker in camera frame
          #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
          #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
          ret_frame_gamma = aruco.estimatePoseSingleMarkers(corners_frame_gamma, marker_size, camera_matrix, camera_distortion)
          #ret_img_gamma = aruco.estimatePoseSingleMarkers(corners_img_gamma, marker_size, camera_matrix, camera_distortion)

          #-- Unpack the output, get only the first
          rvec_frame_gamma, tvec_frame_gamma = ret_frame_gamma[0][0,0,:], ret_frame_gamma[1][0,0,:]
          #rvec_img_gamma, tvec_img_gamma = ret_img_gamma[0][0,0,:], ret_img_gamma[1][0,0,:]

          #-- Draw the detected marker and put a reference frame over it
          aruco.drawDetectedMarkers(frame_gamma, corners_frame_gamma)
          #aruco.drawDetectedMarkers(img_gamma, corners_img_gamma)
          
          aruco.drawAxis(frame_gamma, camera_matrix, camera_distortion, rvec_frame_gamma, tvec_frame_gamma, 10)
          #aruco.drawAxis(img_gamma, camera_matrix, camera_distortion, rvec_img_gamma, tvec_img_gamma, 10)

          #-- Print the tag position in camera frame
          str_position_frame_gamma = "MARKER Position x=%4.1f  y=%4.1f  z=%4.1f"%(tvec_frame_gamma[0], tvec_frame_gamma[1], tvec_frame_gamma[2])
          #str_position_img_gamma, = "MARKER Position x=%4.1f  y=%4.1f  z=%4.1f"%(tvec_img_gamma[0], tvec_img_gamma[1], tvec_img_gamma[2])
          
          cv2.putText(frame_gamma, str_position_frame_gamma, (0, 100), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
          #cv2.putText(img_gamma, str_position_img_gamma, (0, 100), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)          
          
          #-- Obtain the rotation matrix tag->camera
          R_ct_frame_gamma    = np.matrix(cv2.Rodrigues(rvec_frame_gamma)[0])
          #R_ct_img_gamma    = np.matrix(cv2.Rodrigues(rvec_img_gamma)[0])
          
          R_tc_frame_gamma    = R_ct_frame_gamma.T
          #R_tc_img_gamma    = R_ct_img_gamma.T

          #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
          roll_marker_frame_gamma, pitch_marker_frame_gamma, yaw_marker_frame_gamma = rotationMatrixToEulerAngles(R_flip*R_tc_frame_gamma)
          #roll_marker_img_gamma, pitch_marker_img_gamma, yaw_marker_img_gamma= rotationMatrixToEulerAngles(R_flip*R_tc_img_gamma)

          #-- Print the marker's attitude respect to camera frame
          str_attitude_frame = "MARKER Attitude r=%4.1f  p=%4.1f  y=%4.1f"%(math.degrees(roll_marker_frame_gamma),math.degrees(pitch_marker_frame_gamma),
                              90-math.degrees(yaw_marker_frame_gamma))
          
          
          cv2.putText(frame_gamma, str_attitude_frame, (0, 150), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            
        try:
            #--- Display the frame
            cv2.imshow('frame_gamma', frame_gamma)

        except:
            None
    
    except:
        None
        
    #--- use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

