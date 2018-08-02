# coding=gbk
import cv2
import os
import time
import numpy as np
    #设置一些常用的一些参数
    #显示的字体 大小 初始位置等
# 显示ROI为二值模式
# 图像的二值化，就是将图像上的像素点的灰度值设置为0或255，
# 也就是将整个图像呈现出明显的只有黑和白的视觉效果。

#  cv2.threshold  进行阈值化
# 第一个参数  src     指原图像，原图像应该是灰度图
# 第二个参数  x     指用来对像素值进行分类的阈值。
# 第三个参数    y  指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
# 有两个返回值 第一个返回值（得到图像的阈值）   二个返回值 也就是阈值处理后的图像


font = cv2.FONT_HERSHEY_SIMPLEX  # 正常大小无衬线字体
size = 0.5
fx = 10
fy = 355
fh = 18
# ROI框的显示位置
x0 = 300
y0 = 100
# 录制的手势图片大小
width = 200
height = 200
# 每个手势录制的样本数
numofsamples = 1200
counter = 0  # 计数器，记录已经录制多少图片了
# 存储地址和初始文件夹名称
gesturename = ''
path = ''
folder=''# 所有的手势图片都放在里面

# 标识符 bool类型用来表示某些需要不断变化的状态
binaryMode = False  # 是否将ROI显示为而至二值模式
saveImg = False  # 是否需要保存图片

def Color_RGB_Dection(ros):
    img=ros.copy()
    Ker_1 = np.ones((5, 5), np.uint8)
    rows, cols, channels=img.shape
    for r in range(rows):
        for c in range(cols):
            skin = 0
            B = img.item(r, c, 0)
            G = img.item(r, c, 1)
            R = img.item(r, c, 2)
            if((R>90) and (G>40) and (B>20) and ((max(B,G,R)-min(B,G,R))>15)):
                if((R>B)and (R>G) and (abs(R-G)>15)):
                    skin=1
                    img.itemset((r, c, 0), 255)
                    img.itemset((r, c, 1), 255)
                    img.itemset((r, c, 2), 255)
            else:
                if((R>220) and (G>210) and (B>170)):
                   if((R>G) and (R>B) and (abs(R-G)>15)):
                        skin=1
                        img.itemset((r, c, 0), 255)
                        img.itemset((r, c, 1), 255)
                        img.itemset((r, c, 2), 255)
            if 0 == skin:
                img.itemset((r, c, 0), 0)
                img.itemset((r, c, 1), 0)
                img.itemset((r, c, 2), 0)
                # print 'Skin detected!'
    img=cv2.medianBlur(img,11)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Ker_1)
    return img
def Color_YCbCr_Dection(ros):
    img=ros.copy()
    Ker_1 = np.ones((5, 5), np.uint8)
    row,col,channel=img.shape
    Img_YCrCb=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    for r in range(row):
        for c in range(col):
            Cr=Img_YCrCb.item(r,c,1)
            Cb=Img_YCrCb.item(r,c,2)
            if((Cr>=133)and(Cr<=173)and(Cb>=77)and(Cb<=127)):
                img.itemset((r,c,0),255)
                img.itemset((r,c,1),255)
                img.itemset((r,c,2),255)
            else:
                img.itemset((r,c,0),0)
                img.itemset((r,c,1),0)
                img.itemset((r,c,2),0)
    img=cv2.medianBlur(img,13)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Ker_1)
    return img
def binaryMask(frame, x, y, width_1, height_1):
    # 显示方框
    cv2.rectangle(frame, (x, y), (x+width_1, y+height_1), (0, 0, 255))
    #提取ROI像素
    roi = frame[y:y+height_1, x:x+width_1]
    res = Color_YCbCr_Dection(roi)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
    # 保存手势
    if saveImg == True and binaryMode == True:
        saveROI(img)
    elif saveImg == True and binaryMode == False:
        saveROI(roi)
    """这里可以插入代码调用网络"""

    return res

# 保存ROI图像
def saveROI(img):
    global path, counter, gesturename, saveImg
    if counter >=numofsamples:
        # 恢复到初始值，以便后面继续录制手势
        saveImg = False
        gesturename = ''
        counter = 0
        return

    counter += 1
    name = gesturename + str(counter) # 给录制的手势命名
    print("Saving img: ", name)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path+name+'.png', img) # 写入文件
    time.sleep(0.05)
