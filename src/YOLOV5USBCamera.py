# import videocapture as video
import cv2
import time 
from yolov5_model import sampleYOLOV5
from midas_model import MiDaSModel
from acllite_resource import AclLiteResource
import serial
import numpy as np
import array as arr



def find_camera_index():
    max_index_to_check = 10  # Maximum index to check for camera
    for index in range(max_index_to_check):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            return index
    # If no camera is found
    raise ValueError("No camera found.")

if __name__ == '__main__':
    yolov5_model_path = '../model/yolov5s_rgb.om'
    midas_model_path = '../model/model-small.om'
    # midas_width,midas_height = 256,256
    yolov5_model_width = 640
    yolov5_model_height = 640 
    midas_model_type = "small"
    resource = AclLiteResource()
    resource.init()
    yolov5_model = sampleYOLOV5(yolov5_model_path, yolov5_model_width, yolov5_model_height)
    midas_model = MiDaSModel(midas_model_path, resource, midas_model_type)
    yolov5_model.init_resource(resource)

    camera_index = find_camera_index()
    cap = cv2.VideoCapture(camera_index)

    ser=serial.Serial(port="/dev/ttyAMA2",
                  baudrate=115200,
                  bytesize=serial.EIGHTBITS,
                  parity=serial.PARITY_NONE,
                  timeout=0.5
                  )
    
    if ser.isOpen():
            print("Serial is open")
            print(ser.name)
    else:
        print("Serial is not open")

    while True:
        ret, frame = cap.read()
        if not ret:  
            print("Can't receive frame (stream end?). Exiting ...")  
            break  
        #目标检测
        yolov5_frame = frame.copy()
        midas_frame = frame.copy()
        start_time = time.time()
        yolov5_model.preprocess(yolov5_frame)
        yolov5_model.infer()
        trans=yolov5_model.postprocess()
        #深度推理
        midas_model.preprocess(midas_frame)
        midas_model.infer()
        depth_map = midas_model.postprocess()
        fps = 1 / (time.time() - start_time)
        cv2.putText(yolov5_frame,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(depth_map,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.imshow('obj_detection', yolov5_frame)
        cv2.imshow('depth_detection', depth_map)
        #串口发送数据
        

        if(trans!=None):

            rgb_max=np.int32(0)
            tramessage=[0,0,0,0,0]

            for i in range(len(trans)):
                centerx_1=np.uint8((trans[i][0]>>8)&0xff)
                centerx_2=np.uint8(trans[i][0]&0xff)

                rgb=depth_map[trans[i][1],trans[i][0]]
                rgb_sum=(np.int32(rgb[0])<<16)+(np.int32(rgb[1])<<8)+np.int32(rgb[2])

                if(rgb_max<rgb_sum):
                    rgb_max=rgb_sum  
                    tram=arr.array('B', [centerx_1,centerx_2,rgb[0],rgb[1],rgb[2]])
                    for i in range(5):
                        tramessage[i]=tram[i]

            if(rgb_max != 0):
                ser.write(bytes(tramessage))

            print(rgb_max)

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
    cap.release()  
    cv2.destroyAllWindows()
    
    ser.close()

    yolov5_model.release_resource()
    midas_model.release()
