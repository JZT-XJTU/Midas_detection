# import videocapture as video
import numpy as np
import cv2

import time 

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_imageproc import AclLiteImageProc
from acllite_image import AclLiteImage
from label import labels
from acllite_logger import log_error, log_info




class sampleYOLOV5(object):
    def __init__(self, model_path, model_width, model_height):
        self.model_path = model_path
        self.model_width = model_width
        self.model_height = model_height

    def init_resource(self, resource):
        # 初始化 acl resource, create image processor, create model
        self._resource = resource    
        self._dvpp = AclLiteImageProc(self._resource) 
        self._model = AclLiteModel(self.model_path)

    def preprocess(self, frame):
        # 预处理
        self.src_image = frame
        self.resized_image = cv2.resize(frame, (self.model_width, self.model_height))

    def infer(self):
        # 推理
        image_info = np.array([640, 640,
                            640, 640],
                            dtype=np.float32)
        self.result = self._model.execute([self.resized_image, image_info])
    
    def postprocess(self):
        box_num = self.result[1][0, 0]
        box_info = self.result[0].flatten()

        height, width, _ = self.src_image.shape 
        scale_x = width / self.model_width
        scale_y = height / self.model_height

        colors = [0, 0, 255]
        text = ""
        imform =[]
        # draw the boxes in original image
        for n in range(int(box_num)):
            ids = int(box_info[5 * int(box_num) + n])
            score = box_info[4 * int(box_num) + n]
            label = labels[ids] + ":" + str("%.2f" % score)
            top_left_x = box_info[0 * int(box_num) + n] * scale_x
            top_left_y = box_info[1 * int(box_num) + n] * scale_y
            bottom_right_x = box_info[2 * int(box_num) + n] * scale_x
            bottom_right_y = box_info[3 * int(box_num) + n] * scale_y
            cv2.rectangle(self.src_image, (int(top_left_x), int(top_left_y)),
                        (int(bottom_right_x), int(bottom_right_y)), colors)
            p3 = (max(int(top_left_x), 15), max(int(top_left_y), 15))
            position = [int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y)]
            cv2.putText(self.src_image, label, p3, cv2.FONT_ITALIC, 0.6, colors, 1)
            text += f'label:{label} {position}  '

            center_x=(int)((top_left_x+bottom_right_x)/2)
            center_y=(int)((top_left_y+bottom_right_y)/2)
            
            center=[center_x,center_y]

            if(labels[ids] == "person"):
                imform.append(center)

        log_info(text)
        log_info(imform)
        return imform

    def release_resource(self):
        # 释放资源
        del self._resource
        del self._dvpp
        del self._model
        del self.resized_image