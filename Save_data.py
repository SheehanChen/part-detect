#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2018/12/6/006 20:55 
# @Author : Farbror 
# @File : Save_data.py 
# @Software: PyCharm
#本程序用于计算图片的sift特征并使用PCA降维
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
def Get_Imgdata(cwd,classes,data_str,label_str):
    train_img=[]
    train_label=[]
    ii=0
    for index, name in enumerate(classes):
        class_path = os.path.join(cwd, name)
        for img_name in os.listdir(class_path):
            ii=ii+1
            print(ii)
            img_path = class_path + '\\' + img_name
            img=cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            # 找到关键点和描述
            ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # 二值化
            key_query, desc_query = sift.detectAndCompute(binary, None)
            # 把特征点标记到图片上
            img = cv2.drawKeypoints(img, key_query, img)
            pca = PCA(1)  # 保留所有成分
            pca.fit(np.transpose(desc_query))
            low_d = pca.transform(np.transpose(desc_query))
            low_d=low_d.reshape(-1)
            train_img.append(low_d)
            if index ==0:
                val0=[1,0,0,0,0,0]
            elif index==1:
                val0=[0,1,0,0,0,0]
            else:
                if index==2:
                    val0=[0,0,1,0,0,0]
                else:
                    if index==3:
                        val0=[0,0,0,1,0,0]
                    else:
                        if index==4:
                            val0=[0,0,0,0,1,0]
                        else:
                            val0=[0,0,0,0,0,1]
            train_label.append(val0)

    filename =data_str
    np.savetxt(filename,train_img)
    filename = label_str
    np.savetxt(filename,train_label)
    return train_img,train_label
if __name__ == '__main__':
    cwd = "cut_image_merge\\"
    cwd2 = "test image\\"
    classes = ('Flat washer', 'nut', 'screw', 'screw bolt', 'shoulder screw', 'spring washer')
    train_img, train_label = Get_Imgdata(cwd, classes, 'train_data.txt', 'train_label.txt')
    vali_img, vali_label = Get_Imgdata(cwd2, classes, 'val_data.txt', 'val_label.txt')