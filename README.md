# Aruco
本项目基于Opencv-Python，实现Aruco Marker/Aruco Diamand的识别同时测量marker相较于摄像头的位置参数与姿态参数。由于光线对marker的识别影响较大(考虑到某些户外使用的场景)，本项目中也提供了两种不同的解决方案，即**gamma校准**与**摄像头曝光时间**。gamma校准无需额外的硬件设备，但是会有较大的噪声使得图像质量大幅度下滑。而另一种方案则是使用黑白摄像头与灯箱制作marker并控制摄像头的曝光时间。这种方案会带来额外的成本，不过会有更好的效果。
## 参考资料
[单个Aruco码识别](https://github.com/tizianofiorenzani/how_do_drones_work)  
[Aruco Diamand码识别](https://docs.opencv.org/4.2.0/d5/d07/tutorial_charuco_diamond_detection.html)
## 二维码生成
[Aruco Marker](https://chev.me/arucogen/)  
[Aruco Diamand](https://calib.io/pages/camera-calibration-pattern-generator)
