FROM osrf/ros:noetic-desktop-full

ENV DISPLAY host.docker.internal:0.0

# Do not ask the keyboard layout
ARG DEBIAN_FRONTEND=noninteractive

# Install apt-get packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install x11-apps -y
RUN apt-get install -y zsh git vim wget tmux terminator
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set zsh as default shell
ENV SHELL /usr/bin/zsh
RUN zsh
RUN chsh -s /bin/zsh
#zprezto install
RUN git clone --recursive \
    https://github.com/sorin-ionescu/prezto.git \
    $HOME/.zprezto

SHELL ["/bin/zsh", "-c"]
RUN setopt EXTENDED_GLOB; \
    for rcfile in "${ZDOTDIR:-$HOME}"/.zprezto/runcoms/^README.md(.N); do \
    ln -s "$rcfile" "${ZDOTDIR:-$HOME}/.${rcfile:t}"; \
    done
RUN git clone --recurse https://github.com/belak/prezto-contrib $HOME/.zprezto/contrib
RUN sed -i "/'prompt'/c \\\ 'contrib-prompt' \\\\\n  'prompt'" /root/.zpreztorc
RUN sed -i "s/theme 'sorin'/theme 'steeef'/g" /root/.zpreztorc

CMD ["/bin/zsh"]

# Install ROS packages
RUN apt-get update && apt-get install -y \
    ros-noetic-joy ros-noetic-teleop-twist-joy ros-noetic-ackermann-msgs\
    ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
    ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
    ros-noetic-rosserial-python ros-noetic-rosserial-client \
    ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
    ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
    ros-noetic-compressed-image-transport ros-noetic-rqt ros-noetic-rviz \
    ros-noetic-gmapping ros-noetic-navigation ros-noetic-interactive-markers \
    ros-noetic-dynamixel-sdk \
    ros-noetic-novatel-gps-driver\
    ros-noetic-turtlebot3-msgs \
    ros-noetic-turtlebot3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install additional packages
RUN apt-get install apt-transport-https
RUN sh -c 'echo "deb [trusted=yes] https://s3.amazonaws.com/autonomoustuff-repo/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/autonomoustuff-public.list'
RUN apt-get update
RUN apt-get install ros-noetic-pacmod3-msgs  

# Create ROS workspace
RUN echo "source /opt/ros/noetic/setup.zsh" >> /root/.zshrc
RUN mkdir -p /root/catkin_ws/src
RUN cd /root/catkin_ws/src && git clone https://github.com/bsb808/geonav_transform.git
RUN cd /root/catkin_ws && /bin/zsh -c "source /opt/ros/noetic/setup.sh; catkin_make"
RUN echo "source /root/catkin_ws/devel/setup.zsh" >> /root/.zshrc

# Clone the final project repository
RUN cd /root/catkin_ws/src && git clone https://github.com/miki-yuasa/ece484_final_proj.git
RUN apt update && apt-get install -y python3-pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
RUN pip3 install stable-baselines3 gymnasium tensorboard matplotlib scipy numpy seaborn
RUN cd /root/catkin_ws && /bin/zsh -c "catkin_make"
RUN /bin/zsh -c "source /root/catkin_ws/devel/setup.zsh"