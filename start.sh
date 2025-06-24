#!bin/bash

docker build -t ros2_naoqi .

NAO_IP=169.254.39.170

# qicli call ALAutonomousLife.setState disabled
# qicli call ALMotion.wakeUp

docker run -it --rm --net host --name Nao -e NAO_IP=$NAO_IP ros2_naoqi:latest

