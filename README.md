# Midas_detection（昇腾/香橙派 AI Pro）

此项目为JZT在电子系统设计与创新课程的课程设计项目所负责的内容

基于昇腾平台（OrangePi AI Pro）的实时视觉融合项目：
- 使用 **YOLOv5** 进行目标检测
- 使用 **MiDaS** 进行单目深度估计
- 将检测结果与深度图融合，并通过串口输出目标信息

适用于电设/嵌入式视觉场景中的“检测 + 距离感知”任务。

## 效果展示

![效果展示](results_show/效果展示.jpg)

---

## 1. 项目特点

- **端侧实时推理**：OM 模型在昇腾 NPU 上运行
- **双模型融合**：检测框中心点映射到深度伪彩色图进行距离指示
- **串口联动**：向下位机发送目标中心与深度颜色值
- **可视化输出**：同时显示目标检测窗口与深度估计窗口

---

## 2. 目录结构

```text
Midas_detection/
├── data/
├── model/
│   ├── yolov5s_rgb.om
│   ├── model-small.om
│   ├── model-large.om
│   ├── aipp_*.cfg
│   └── *.onnx
└── src/
    ├── YOLOV5USBCamera.py      # 主程序（摄像头采集、推理、融合、串口发送）
    ├── yolov5_model.py         # YOLOv5 推理封装
    ├── midas_model.py          # MiDaS 推理封装
    ├── midas_transforms.py     # MiDaS 预处理
    ├── acllite_*.py            # ACLLite 资源与推理工具
    └── label.py                # 类别标签
```

---

## 3. 环境要求

### 3.1 硬件

- 香橙派 AI Pro（昇腾 NPU）
- USB 摄像头
- （可选）串口外设（默认 `/dev/ttyAMA2`）

### 3.2 软件

- Linux（板端系统）
- Python 3.8+
- CANN / Ascend Toolkit（与板端驱动匹配）

### 3.3 Python 依赖（最小集合）

```bash
pip3 install numpy opencv-python Pillow pyserial av==6.2.0
```

> 说明：`acllite` 与 `acl` 依赖 Ascend CANN 环境，请先正确安装驱动与工具链。

---

## 4. 昇腾环境变量配置

请根据你的实际安装路径调整：

```bash
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
export THIRDPART_PATH=${DDK_PATH}/thirdpart
export PYTHONPATH=${THIRDPART_PATH}/python:$PYTHONPATH
```

如需使用 ACLLite（源码方式），可将 ACLLite python 目录拷贝到 `${THIRDPART_PATH}` 下。

---

## 5. 模型说明

当前主程序默认使用：
- YOLOv5：`model/yolov5s_rgb.om`（输入 640×640）
- MiDaS：`model/model-small.om`（输入 256×256）

在 [src/YOLOV5USBCamera.py](src/YOLOV5USBCamera.py) 中可修改：
- `yolov5_model_path`
- `midas_model_path`
- `midas_model_type`（`small` 或 `large`）

---

## 6. 运行方法

在项目根目录执行：

```bash
cd src
python3 YOLOV5USBCamera.py
```

运行后会打开两个窗口：
- `obj_detection`：目标检测结果
- `depth_detection`：深度伪彩色结果

按 `q` 退出。

---

## 7. 融合与串口输出逻辑

主流程位于 [src/YOLOV5USBCamera.py](src/YOLOV5USBCamera.py)：

1. YOLOv5 检测图像中的目标
2. 仅保留 `person` 类目标中心点（见 [src/yolov5_model.py](src/yolov5_model.py)）
3. 在 MiDaS 深度伪彩色图中读取对应像素颜色
4. 选择“颜色值综合最大”的目标
5. 通过串口发送 5 字节数据：
   - Byte0~1：中心点 `x` 坐标高低字节
   - Byte2~4：深度伪彩色 `B,G,R`

默认串口参数：
- 端口：`/dev/ttyAMA2`
- 波特率：`115200`
- 8N1，超时 `0.5s`

---


## 9. 致谢

- MiDaS 单目深度估计开源方案：https://github.com/isl-org/MiDaS
- YOLOv5 目标检测模型
- 华为 Ascend ACLLite 推理样例与工具链
