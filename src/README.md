# ACLLite-python快速部署

## 安装步骤

设置环境变量，配置程序编译依赖的头文件，库文件路径。“/usr/local/Ascend”请替换“Ascend-cann-toolkit”包的实际安装路径。

   ```
    export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
    export THIRDPART_PATH=${DDK_PATH}/thirdpart
    export PYTHONPATH=${THIRDPART_PATH}/python:$PYTHONPATH
   ```

  创建THIRDPART_PATH路径
   ```
    mkdir -p ${THIRDPART_PATH}
   ```
运行环境安装python-acllite所需依赖

   ```
    # 安装ffmpeg
    sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev
    # 安装其它依赖
    python3 -m pip install --upgrade pip
    python3 -m pip install Cython
    sudo apt-get install pkg-config libxcb-shm0-dev libxcb-xfixes0-dev
    # 安装pyav
    python3 -m pip install av==6.2.0
    # 安装pillow 的依赖
    sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk
    # 安装numpy和PIL
    python3 -m pip install numpy
    python3 -m pip install Pillow
   ```
   
  python acllite库以源码方式提供，安装时将acllite目录拷贝到运行环境的第三方库目录

   ```
    # 将acllite目录拷贝到第三方文件夹中。后续有变更则需要替换此处的acllite文件夹
    cp -r ${HOME}/ACLLite/python ${THIRDPART_PATH}
   ```
