#!bin/bash

docker build -t ros2_naoqi .

NAO_IP=169.254.211.244

# qicli call ALAutonomousLife.setState disabled
# qicli call ALMotion.wakeUp

docker run -it --rm --net host --name Nao -e NAO_IP=$NAO_IP -e OPENAI_API_KEY=$(cat "$HOME/Documents/key.txt") ros2_naoqi:latest

