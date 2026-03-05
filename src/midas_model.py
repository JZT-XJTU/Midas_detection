import cv2
import numpy as np
# import torch
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_imageproc import AclLiteImageProc
from midas_transforms import Resize, NormalizeImage, PrepareForNet

class MiDaSModel(object):
    """MiDaS单目深度估计OM模型处理类"""
    def __init__(self, model_path, resource, model_type):
        self.model_path = model_path
        self.model_type = model_type
        self._resource = resource
        self._dvpp = AclLiteImageProc(self._resource) 
        
        # 根据模型类型设置输入尺寸
        if model_type == "large":
            self.net_w, self.net_h = 384, 384
        elif model_type == "small":
            self.net_w, self.net_h = 256, 256
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 初始化预处理流程
        self.resize = Resize(
            self.net_w,
            self.net_h,
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        
        def compose2(f1, f2):
            return lambda x: f2(f1(x))
        self.transform = compose2(self.resize, PrepareForNet())

        """初始化模型（依赖已创建的ACL资源）"""
        self._model = AclLiteModel(self.model_path)

    def preprocess(self, frame):
        """预处理输入图像"""
        self.src_image = frame
        # 转换为RGB并归一化
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        # 应用预处理
        img_input = self.transform({"image": img_rgb})["image"]
        # 调整维度为(1, 3, H, W)并转换为float32
        self.input_data = img_input.reshape(1, 3, self.net_h, self.net_w).astype(np.float32)

    def infer(self):
        """执行推理"""
        self.result = self._model.execute([self.input_data])[0]

    def postprocess(self):
        """后处理得到深度图"""
        # 重塑输出并调整为原图尺寸
        prediction = np.array(self.result).reshape(self.net_h, self.net_w)
        self.depth_map = cv2.resize(
            prediction, 
            (self.src_image.shape[1], self.src_image.shape[0]), 
            interpolation=cv2.INTER_CUBIC
        )
        # 归一化深度图用于可视化
        depth_min = self.depth_map.min()
        depth_max = self.depth_map.max()
        self.depth_vis = (255 * (self.depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        # 转换为伪彩色图增强可视化效果
        self.depth_vis = cv2.applyColorMap(self.depth_vis, cv2.COLORMAP_JET)
        return self.depth_vis

    def release(self):
        """释放模型资源"""
        del self._model
        del self._resource
        del self._dvpp
        del self.resize