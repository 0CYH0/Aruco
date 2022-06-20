#encoding:utf-8

import cv2 as cv
import numpy as np

# 创建ChArUco标定板
dictionary = cv.aruco.getPredefinedDictionary(dict=cv.aruco.DICT_6X6_250)
board = cv.aruco.CharucoBoard_create(squaresY=3,
                                     squaresX=3,
                                     squareLength=0.04,
                                     markerLength=0.02,
                                     dictionary=dictionary)
img_board = board.draw(outSize=(600, 500), marginSize=10, borderBits=1)
cv.imwrite(filename='F:/Teacher_Smart_Car_Project/Aruco_diamond/Aruco_diamond_marker/charuco.png', img=img_board, params=None)

camera_matrix = np.array([[636.79414097, 0, 638.12408277],
                          [0, 636.21220437, 373.77408497],
                          [0,            0,          1]])
dist_coefs = np.array([-0.02062563,  0.08896318,  0.00069723,  0.00023204, -0.08330528])

Ids_to_find = np.array([[0], [1], [2], [3]]) # 定义指定的识别ID号 

# 主要用于图形的绘制与显示
img_color = cv.cvtColor(src=img_board,
                        code=cv.COLOR_GRAY2BGR,
                        dstCn=None)

# 查找标志块的左上角点
corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(image=img_board,
                                                         dictionary=dictionary,
                                                         parameters=None,
                                                         cameraMatrix=camera_matrix,
                                                         distCoeff=dist_coefs)
# print(corners)
# print(ids)
if len(ids) != None:
    print((int(corners[0][0][0][0]), int(corners[0][0][0][1])))
    print((int(corners[1][0][0][0]), int(corners[1][0][0][1])))
    cv.circle(img_color, (int(corners[0][0][0][0]), int(corners[0][0][0][1])), 8, [0, 255, 0])
    cv.circle(img_color, (int(corners[1][0][0][0]), int(corners[1][0][0][1])), 8, [0, 255, 0])
    cv.circle(img_color, (int(corners[2][0][0][0]), int(corners[2][0][0][1])), 8, [0, 255, 0])
    # 绘制标志块的左上角点与对应的ID
    cv.aruco.drawDetectedMarkers(image=img_color, corners=corners, ids=ids, borderColor=None)
    cv.imshow("out0", img_color)
    cv.waitKey()

    # 棋盘格黑白块内角点
    retval, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                            markerIds=ids,
                                                                            image=img_board,
                                                                            board=board,
                                                                            cameraMatrix=camera_matrix,
                                                                            distCoeffs=dist_coefs)
    print(charucoCorners)
    print(charucoIds)
    
    if len(charucoIds) != None and all(charucoIds == Ids_to_find):
        # 绘制棋盘格黑白块内角点
        cv.aruco.drawDetectedCornersCharuco(img_color, charucoCorners, charucoIds, [0, 0, 255])
        cv.imshow("out1", img_color)
        cv.waitKey()

        rvec = None
        tvec = None
        retval, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix,
                                                               dist_coefs, rvec, tvec)
        if retval:
            cv.aruco.drawAxis(img_color, camera_matrix, dist_coefs, rvec, tvec, 0.01)

cv.imshow("out2", img_color)
cv.waitKey()