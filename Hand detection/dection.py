# coding=gbk
import cv2
import os
import time
import numpy as np
    #����һЩ���õ�һЩ����
    #��ʾ������ ��С ��ʼλ�õ�
# ��ʾROIΪ��ֵģʽ
# ͼ��Ķ�ֵ�������ǽ�ͼ���ϵ����ص�ĻҶ�ֵ����Ϊ0��255��
# Ҳ���ǽ�����ͼ����ֳ����Ե�ֻ�кںͰ׵��Ӿ�Ч����

#  cv2.threshold  ������ֵ��
# ��һ������  src     ָԭͼ��ԭͼ��Ӧ���ǻҶ�ͼ
# �ڶ�������  x     ָ����������ֵ���з������ֵ��
# ����������    y  ָ������ֵ���ڣ���ʱ��С�ڣ���ֵʱӦ�ñ�������µ�����ֵ
# ����������ֵ ��һ������ֵ���õ�ͼ�����ֵ��   ��������ֵ Ҳ������ֵ������ͼ��


font = cv2.FONT_HERSHEY_SIMPLEX  # ������С�޳�������
size = 0.5
fx = 10
fy = 355
fh = 18
# ROI�����ʾλ��
x0 = 300
y0 = 100
# ¼�Ƶ�����ͼƬ��С
width = 200
height = 200
# ÿ������¼�Ƶ�������
numofsamples = 1200
counter = 0  # ����������¼�Ѿ�¼�ƶ���ͼƬ��
# �洢��ַ�ͳ�ʼ�ļ�������
gesturename = ''
path = ''
folder=''# ���е�����ͼƬ����������

# ��ʶ�� bool����������ʾĳЩ��Ҫ���ϱ仯��״̬
binaryMode = False  # �Ƿ�ROI��ʾΪ������ֵģʽ
saveImg = False  # �Ƿ���Ҫ����ͼƬ

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
    # ��ʾ����
    cv2.rectangle(frame, (x, y), (x+width_1, y+height_1), (0, 0, 255))
    #��ȡROI����
    roi = frame[y:y+height_1, x:x+width_1]
    res = Color_YCbCr_Dection(roi)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
    # ��������
    if saveImg == True and binaryMode == True:
        saveROI(img)
    elif saveImg == True and binaryMode == False:
        saveROI(roi)
    """������Բ�������������"""

    return res

# ����ROIͼ��
def saveROI(img):
    global path, counter, gesturename, saveImg
    if counter >=numofsamples:
        # �ָ�����ʼֵ���Ա�������¼������
        saveImg = False
        gesturename = ''
        counter = 0
        return

    counter += 1
    name = gesturename + str(counter) # ��¼�Ƶ���������
    print("Saving img: ", name)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path+name+'.png', img) # д���ļ�
    time.sleep(0.05)
