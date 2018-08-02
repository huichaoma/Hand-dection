import os
import numpy as np
from sklearn import model_selection, utils
from PIL import Image
import cv2

PATH='D:\\Interns software\\code\\detection\\Test_Picture\\'
img_rows = 200
img_cols = 200
img_channels = 1
nb_classes = 10 # 类别

def modlist(path):
    # 列出path里面所有文件信息
    retlist = []
    listing = os.listdir(path)
    for name in listing:
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist
def Picture_Size(path):
    List_P=os.listdir(path)
    for i in range(0,len(List_P)):
        file=path+List_P[i]
        if os.path.isfile(file):
            Ros=cv2.imread(file)
            Ros = cv2.resize(Ros, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
            res = cv2.cvtColor(Ros, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(file,img)

def Initializer():
    # 初始化数据，产生训练测试数据和标签
    #Picture_Size(PATH)
    imlist= modlist(PATH)
    total_images = len(imlist) # 样本数量
    immatrix = np.array([np.array(Image.open(PATH+image).convert('L')).flatten() for image in imlist], dtype='float32')
    # 注 PIL 中图像共有9中模式 模式“L”为灰色图像 0黑 255白
    # 转换公式 L = R * 299/1000 + G * 587/1000+ B * 114/1000
    # 开始创建标签
    label = np.ones((total_images, ), dtype=int)
    samples_per_class = total_images / nb_classes # 每类样本的数量，（由于录制的时候录制的一样多，这里可以这样写，如果不一样多，标签就需要根据文件名来进行获取）
    s = 0
    r = samples_per_class
    # 开始赋予标签（01234）
    for index in range(nb_classes):
        # 0-1200: 0
        # 1200-2400:1
        #...
        label[int(s):int(r)] = index
        #print(imlist[index*1200])
        s = r
        r = s + samples_per_class
    data, label = utils.shuffle(immatrix, label)#混乱器
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, label, test_size=0.1, random_state=4)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels) #  tensorflow的图像格式为[batch, W, H, C]
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels) #  tensorflow的图像格式为[batch, W, H, C]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, X_test, y_train, y_test
