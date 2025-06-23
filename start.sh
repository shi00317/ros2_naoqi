#!bin/bash

docker build -t ros2_naoqi .

NAO_IP=127.0.0.1
docker run -it --rm --net host --name Nao -e NAO_IP=$NAO_IP ros2_naoqi:latest