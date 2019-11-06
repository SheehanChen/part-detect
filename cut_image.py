# -*- coding: utf-8 -*-
'''
本程序用于切除图片中的空白
'''

import cv2
import numpy as np
import os
import datetime

def measure_object(image):
    '''
    函数用于把零件的空白区域切除
    :param image:输入把标定完边缘轮廓切好的轮廓
    :return:返回切除好的图像
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # 二值化
    outImage, contours, hireachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 发现轮廓
    area = []
    for i in range(len(contours)):  # 最大连通区域
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    rect = cv2.minAreaRect(contours[max_idx])  # 最小外接矩形
    box = cv2.boxPoints(rect)  # 取外接矩形四个点坐标
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(box)  # 确定切除图像的边界
    if y<0:  # 处理边缘情况
        y = 0
    if y+h>1060:
        y = 1060 - h
    if x<0:
        x = 0
    if x+w>1890:
        x = 1890 - w
    newimage = image[y:y + h, x:x + w]  # 先用y确定高，再用x确定宽
    return newimage
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    filelist =os.path.join("/home/chenxin/Document/workspace/machine_recognization_version_5/calibration")
    # 遍历该路径下的文件夹的子文件夹的图片
    for infile in os.listdir(filelist):
        os.mkdir("part recognize\\cut_image\\" + infile)
        filelist_2 = os.path.join("machine_recognization_version_5\\calibration\\"+infile)
        for infile_in in os.listdir(filelist_2):
            img_ol = cv2.imread("machine_recognization_version_5\\"
                                "calibration\\" + infile + "\\" + infile_in)
            img = img_ol[10:1070, 10:1900]  # 切除标定后的黑色轮廓
            newimage = measure_object(img)
            cv2.imwrite("python_version\\cut_image\\" + infile + "\\" + infile_in, newimage)
            # 保存图像
    endtime = datetime.datetime.now()
    print("start time is:", starttime,"\n")  # 输出程序执行时间
    print("end time is :", endtime,"\n")
    print(endtime - starttime).second
