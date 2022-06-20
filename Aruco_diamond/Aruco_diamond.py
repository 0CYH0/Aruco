#encoding:utf-8

import cv2 as cv
import numpy as np
import sys, time, math

#--- 180 deg rotation matrix around the x axis
R_flip      = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

camera_matrix = np.array([[636.79414097, 0, 638.12408277],
                          [0, 636.21220437, 373.77408497],
                          [0,            0,            1]])
dist_coefs = np.array([-0.02062563,  0.08896318,  0.00069723,  0.00023204, -0.08330528])
Ids_to_find = np.array([0, 1 , 2 , 3]) # 定义指定的识别ID号

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

def main():
    # 创建ChArUco标定板
    dictionary = cv.aruco.getPredefinedDictionary(dict=cv.aruco.DICT_6X6_250)
    parameters  = cv.aruco.DetectorParameters_create()

    board = cv.aruco.CharucoBoard_create(squaresY=3,
                                        squaresX=3,
                                        squareLength=0.2,
                                        markerLength=0.12,
                                        dictionary=dictionary)
    img_board = board.draw(outSize=(600, 500), marginSize=10, borderBits=1)
    #cv.imwrite(filename='F:/Teacher_Smart_Car_Project/Aruco_diamond/Aruco_diamond_marker/charuco.png', img=img_board, params=None)

    #--- Capture the videocamera (this may also be a video or a picture)
    cap = cv.VideoCapture(1)

    #-- Set the camera size as the one it was calibrated with
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    #-- Font for the text in the image
    font = cv.FONT_HERSHEY_PLAIN

    # 主要用于图形的绘制与显示
    img_color = cv.cvtColor(src=img_board,
                            code=cv.COLOR_GRAY2BGR,
                            dstCn=None)

    while True:
        
        #-- Read the camera frame
        ret, frame = cap.read()   
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 查找标志块的左上角点
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(image = frame_gray,
                                                                dictionary = dictionary,
                                                                parameters = parameters,
                                                                cameraMatrix = camera_matrix,
                                                                distCoeff = dist_coefs)
        
        if ids is not None:  
            charucoCorners, charucoIds = cv.aruco.detectCharucoDiamond(frame_gray, corners, ids,
                                                                        squareMarkerLengthRate = 2,
                                                                        diamondCorners = corners,
                                                                        diamondIds = ids,
                                                                        cameraMatrix = camera_matrix,
                                                                        distCoeffs = dist_coefs)# squareMarkerLengthRate = squareLength/markerLength

            ids = ids.reshape(-1)       # 数组降维
            ids = ids[np.argsort(ids)]  # 升序排列
            
            if len(ids) == len(Ids_to_find):    #先判断元素个数是否相同
                
                if (ids == Ids_to_find).any():  #若元素个数相同则判断四个独立的Aruco码都被识别到了

                    #cv.aruco.drawDetectedMarkers(frame, corners , ids)        
                    cv.aruco.drawDetectedDiamonds(frame, charucoCorners, charucoIds)
                    
                    #-- ret = [rvec, tvec, ?]
                    #-- 摄像机帧中每个标记的旋转和位置数组
                    #-- rvec = [[rvec_1], [rvec_2], ...]    标记器相对于相机框架的姿态
                    #-- tvec = [[tvec_1], [tvec_2], ...]    标记在相机框中的位置       
                    ret_frame = cv.aruco.estimatePoseSingleMarkers(charucoCorners, 6.4, camera_matrix, dist_coefs)
                    rvec_frame, tvec_frame = ret_frame[0][0,0,:], ret_frame[1][0,0,:]
                    
                    cv.aruco.drawAxis(frame, camera_matrix, dist_coefs, rvec_frame, tvec_frame, 10)
                    
                    #-- Print the tag position in camera frame
                    str_position_frame = "MARKER Position x=%4.1f  y=%4.1f  z=%4.1f"%(tvec_frame[0], tvec_frame[1], tvec_frame[2])          
                    cv.putText(frame, str_position_frame, (0, 100), font, 1.5, (0, 0, 255), 2, cv.LINE_AA)
                    #print("%4.4f"%(tvec_frame[2]))
                    
                    #-- Obtain the rotation matrix tag->camera
                    R_ct_frame    = np.matrix(cv.Rodrigues(rvec_frame)[0])                
                    R_tc_frame    = R_ct_frame.T

                    #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
                    roll_marker_frame, pitch_marker_frame, yaw_marker_frame = rotationMatrixToEulerAngles(R_flip*R_tc_frame)

                    #-- Print the marker's attitude respect to camera frame
                    str_attitude_frame = "MARKER Attitude r=%4.1f  p=%4.1f  y=%4.1f"%(math.degrees(roll_marker_frame),math.degrees(pitch_marker_frame),
                                        90-math.degrees(yaw_marker_frame))               
                    cv.putText(frame, str_attitude_frame, (0, 150), font, 1.5, (0, 0, 255), 2, cv.LINE_AA)
                
        try:
            #--- Display the frame
            cv.imshow('frame', frame)

        except:
            None

        #--- use 'q' to quit
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

if __name__ == '__main__':
    main()

