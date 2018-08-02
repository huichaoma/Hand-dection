import dection

import training
import CNN
import numpy as np
import cv2
import time
import threading
import os

path_m = 'D:\\Interns software\\code\\detection\\'
Gesture=[ ]
exitFlag=True
Prediction_Flag=False
Prediction_Perfect=False
cap = cv2.VideoCapture(0)  # 0为（笔记本）内置摄像头

class mythread(threading.Thread):
    def __init__(self,threadID,name):
        threading.Thread.__init__(self)
        self.threadID=threadID
        self.name=name
    def run(self):
        global Prediction_Flag,Prediction_Perfect,exitFlag
        while exitFlag:
            if Prediction_Flag:
                print("Strat Prediction")
                for i in range(10):
                    threadLock.acquire()
                    ret_1,frame_1=cap.read()
                    threadLock.release()
                    time.sleep(0.1)
                    if ret_1:
                        frame_1 = cv2.flip(frame_1, 1)  # 第二个参数大于0：就表示是沿y轴翻转
                        Box = dection.binaryMask(frame_1, dection.x0, dection.y0, 300, 300)
                        Box_P = cv2.resize(Box, (dection.width, dection.height), interpolation=cv2.INTER_CUBIC)
                        Box_P = np.reshape(Box_P, [dection.width, dection.height, 1])
                        Gesture.append(CNN.Gussgesture(Box_P))
                    else:
                        print('Predictive thread read failed, unpredictable！')
                Prediction_Perfect=True
                Prediction_Flag=False
            key=cv2.waitKey(5)&0xFF
            if(key==27):
                break
        cv2.destroyWindow('Thread_box')

if __name__=='__main__':
    threadLock = threading.Lock()
    thread1 = mythread(1, "Thread_1")
    thread1.start()
    exitFlag = True
    Temp_Gesture=''

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # 第二个参数大于0：就表示是沿y轴翻转
            roi = dection.binaryMask(frame,dection.x0,dection.y0, 300, 300)
        else:
            print('Camera read failed！')
        # 显示提示语
        cv2.putText(frame, "Option: ", (dection.fx, dection.fy), dection.font, dection.size, (0, 0, 255))  # 标注字体
        cv2.putText(frame, "b-'Binary mode'/ r- 'RGB mode' ", (dection.fx, dection.fy + dection.fh), dection.font, dection.size, (0, 0, 255))  # 标注字体
        cv2.putText(frame, "p-'prediction mode'", (dection.fx, dection.fy + 2 * dection.fh), dection.font, dection.size, (0, 0, 255))  # 标注字体
        cv2.putText(frame, "s-'new gestures(twice)'", (dection.fx, dection.fy + 3 * dection.fh), dection.font, dection.size, (0, 0, 255))  # 标注字体
        cv2.putText(frame, "q-'quit'", (dection.fx, dection.fy + 4 * dection.fh), dection.font, dection.size, (0, 0, 255))  # 标注字体

        key = cv2.waitKey(5) & 0xFF  # 等待键盘输入，
        if key == ord('b'):  # 将ROI显示为二值模式
            dection.binaryMode = True
            print('Binary Threshold filter active')
        elif key == ord('r'):  # RGB模式
            dection.binaryMode = False
            print('RGB Model active')
        if key == ord('i'):  # 调整ROI框
            dection.y0 = dection.y0 - 5
        elif key == ord('k'):
            dection.y0 = dection.y0 + 5
        elif key == ord('j'):
            dection.x0 = dection.x0 - 5
        elif key == ord('l'):
            dection.x0 = dection.x0 + 5

        if key == ord('p'):
            """调用模型开始预测"""
            Prediction_Flag=True
        if key == ord('q'):
            exitFlag = False
            break

        if key == ord('s'):
            """录制新的手势（训练集）"""
            # saveImg = not saveImg # True
            if dection.gesturename != '':  #
                dection.saveImg = True
            else:
                print("Enter a gesture group name first, by enter press 'n'! ")
                dection.saveImg = False
        elif key == ord('n'):
            # 开始录制新手势
            # 首先输入文件名字
            dection.folder = (input("enter the gesture folder name: "))
            if (not os.path.exists(dection.folder)):
                os.makedirs(dection.folder)
            dection.path = path_m + dection.folder + '\\'
            dection.gesturename = (input("enter the gesture file name: "))

        # 展示处理之后的视频帧
        cv2.imshow('Main', frame)
        if Prediction_Perfect:
            Prediction_Perfect=False
            Temp_Gesture = Gesture[max(Gesture.count(x) for x in range(len(Gesture)))]
            Gesture.clear()
            print("预测手势： %s" % Temp_Gesture)
        if Temp_Gesture:
            cv2.putText(frame, Temp_Gesture, (480, 440), dection.font, 1, (0, 0, 255))  # 标注字体
        if (dection.binaryMode):
            cv2.imshow('ROI', roi)
        else:
            cv2.imshow("ROI", frame[dection.y0:dection.y0 + 300, dection.x0:dection.x0 + 300])
    # 最后记得释放捕捉
    cap.release()
    cv2.destroyWindow('Main')
    cv2.destroyWindow('ROI')