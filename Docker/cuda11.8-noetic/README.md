# Dockerfile

11.8.0-cudnn8-devel-ubuntu20.04 + ros noetic版本

1. 下载基础镜像

   ```
   docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
   ```

2. 编译镜像

   --add-host是为了翻墙

   ```
   docker build -f Dockerfile.cuda118 -t cuda-ros:11.8-noetic --add-host raw.githubusercontent.com:199.232.28.133 .
   ```

3. 测试镜像

   ```
   docker run -t -i --gpus all cuda-ros:11.8-noetic bash
   
   nvcc -V
   nvidia-smi
   ```


# Dockerfile.cuda

在上面的基础上安装conda, zsh等开发工具