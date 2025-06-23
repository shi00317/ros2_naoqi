#!bin/bash

ssh-keygen -t rsa -b 4096 -f ./my_docker_ssh_key -N ""

ssh-copy-id -i ./my_docker_ssh_key.pub nao@169.254.58.79
