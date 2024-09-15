FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS cuda-env

# 防止在非交互模式下的TZ数据错误
ARG DEBIAN_FRONTEND=noninteractive

# 更新并安装基本工具
RUN apt-get update && apt-get install -y \
    lsb-release \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 添加ROS Noetic的源
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 添加ROS key
RUN curl -sSL 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc' | apt-key add -

# 安装ROS Noetic Desktop-Full
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

# 初始化rosdep
RUN apt-get update && apt-get install -y \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/* \
    && rosdep init && rosdep update

# 设置环境变量
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# 安装其他常用工具
RUN apt-get update && apt-get install -y \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

## 设置默认工作目录
#WORKDIR /root
#
## CUDA
#RUN echo "export PATH=\$PATH:\$HOME/.local/bin:\$HOME/.local/bin" >> $HOME/.bashrc
#RUN echo "export PATH=\$PATH:/usr/local/cuda-11.8/bin" >> $HOME/.bashrc
#RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64/" >> $HOME/.bashrc
#RUN echo "export CUDA_HOME=/usr/local/cuda" >> $HOME/.bashrc

CMD ["/bin/bash"]