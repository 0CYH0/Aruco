#!/bin/sh


while true

do

ps -ef | grep "aruco123" | grep -v "grep"

if [ "$?" -eq 1 ]

then

python3 /home/pi/Desktop/aruco/aruco.py
 #启动应用，修改成自己的启动应用脚本或命令

echo "process has been restarted!"

else

echo "process already started!"

fi

sleep 0.1

done
