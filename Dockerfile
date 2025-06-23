FROM ros:iron

RUN apt update -y \
    && apt install -y \
        tmux \
        nano \
        vim \
        ros-${ROS_DISTRO}-naoqi-libqi \
        ros-${ROS_DISTRO}-naoqi-libqicore \
        ros-${ROS_DISTRO}-naoqi-bridge-msgs \
        ros-${ROS_DISTRO}-nao-meshes \
        ros-${ROS_DISTRO}-naoqi-driver \
    && apt autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY naoqi_start.bash /naoqi_start.bash
COPY boot_config.json /opt/ros/${ROS_DISTRO}/share/naoqi_driver/share/boot_config.json

RUN chmod +x /naoqi_start.bash
ENTRYPOINT ["/naoqi_start.bash"]
