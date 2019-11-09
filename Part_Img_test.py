#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2018/11/12/012 15:45 
# @Author : Farbror 
# @File : Part_Img_test.py 
# @Software: PyCharm
'''

    这个版本加入了对零件尺寸的判断

'''

import datetime
import os
import cv2
import tensorflow as tf
from sklearn.decomposition import PCA
# import Cut_Img as ct
import numpy as np
rig = 0


def flat_washer(length):
    if length < 265.0 or length > 320.0:
        return '_None'
    elif length < 296.5:
        return '_M22_39_3'
    else:
        return '_M24_44_4'


def nut(length):
    if length < 175.0 or length > 345.0:
        return '_None'
    elif length < 251.5:
        return '_1_8'
    else:
        return '_9_16_12'


def spring_washer(length):
    if length < 220.0 or length > 280.0:
        return '_None'
    elif length < 248.6:
        return '_M22'
    else:
        return '_M24'


def screw_bolt(length):
    if length < 485.0 or length > 960.0:
        return '_None'
    elif length < 520.0:
        return '_M12_60'
    elif length < 555.3:
        return '_M12_65'
    elif length < 613.0:
        return '_M12_70'
    elif length < 684.0:
        return '_M12_80'
    elif length < 758.3:
        return '_M12_90'
    elif length < 833.1:
        return '_M12_100'
    elif length < 903.6:
        return '_M12_110'
    else:
        return '_M12_120'


def shoulder_screw(length):
    if length < 285.0 or length > 810.0:
        return '_None'
    if length < 314.8:
        return '_M8_20'
    elif length < 351.4:
        return '_M8_25'
    elif length < 386.9 :
        return '_M8_30'
    elif length < 421.9 :
        return '_M8_35'
    elif length < 457.8:
        return '_M8_40'
    elif length < 496.5:
        return '_M8_45'
    elif length < 549.2:
        return '_M8_50'
    elif length < 602.5:
        return '_M8_60'
    elif length < 656.5:
        return '_M8_65'
    elif length < 714.0:
        return '_M8_75'
    elif length < 748.4:
        return '_M8_80'
    elif length < 776.4:
        return '_M8_85'
    else:
        return '_M8_90'


def screw(length):
    if length < 135.0 or length > 380.0:
        return '_None'
    if length < 166.0:
        return '_M6_20'
    elif length < 202.1:
        return '_M6_25'
    elif length < 238.2:
        return '_M6_30'
    elif length < 274.3:
        return'_M6_35'
    elif length < 311.4:
        return'_M6_40'
    elif length < 347.9:
        return '_M6_45'
    else:
        return '_M6_50'


def foo1(prenum,l):
    return {
        0: "Flat washer"+flat_washer(l),
        1: "nut"+nut(l),
        2: "screw"+screw(l),
        3: "screw_bolt"+screw_bolt(l),
        4: "shoulder screw"+shoulder_screw(l),
        5: "spring washer"+spring_washer(l),
    }.get(prenum, 'error')  # 'error'为默认返回值，可自设置


def Predict_img(dir):
    img = cv2.imread(dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sift = cv2.xfeatures2d.SIFT_create()
    # 找到关键点和描述
    key_query, desc_query = sift.detectAndCompute(binary, None)
    # 把特征点标记到图片上
    pca = PCA(1)  # 保留所有成分
    pca.fit(np.transpose(desc_query))
    low_d = pca.transform(np.transpose(desc_query))
    low_d = low_d.reshape(-1)
    saver = tf.train.import_meta_graph("./5_2model/model.ckpt.meta")
    graph = tf.get_default_graph()
    # 通过 Tensor 名获取变量
    x = graph.get_tensor_by_name("x_input:0")
    y_ = graph.get_tensor_by_name("y_input:0")
    y = tf.get_collection("pred_network")
    temp=[low_d]
    with tf.Session() as sess:
        saver.restore(sess, "./5_2model/model.ckpt")
        output = sess.run(y, feed_dict={x: temp})
        prenum = np.argmax(output[0])

    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        l = max(rect[1])
    return foo1(prenum, l)


def Op_result(dir):
    num=0
    rig=0
    # 遍历文件夹，输出文件夹及子文件夹
    if os.path.exists(dir):
        path_dir = os.path.abspath(dir)
        for i in os.listdir(path_dir):
            path_i = os.path.join(path_dir, i)
            for img_name in os.listdir(path_i):
                num=num+1
                img_path = path_i + '/' + img_name
                ans = Predict_img(img_path)
                imgname_split = img_name[0:(img_name.find('_calibration'))]
                if ans == imgname_split:
                    rig = rig+1
                else:
                    print("the picture " + img_name + " is " + i + "  ", end="")
                    print("图片上的零件被识别为" + ans + '\n')
    print("测试照片总数为"+str(num)+" 识别正确共"+str(rig)+"张")
    print("测试图片正确率为:"+str(rig/num*100)+"%")


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    cwd2 = "test image"
    classes = ('Flat washer', 'nut', 'screw', 'screw bolt', 'shoulder screw', 'spring washer')
    Op_result(cwd2)
    endtime = datetime.datetime.now()
    print("start time is:", starttime,"\n")#输出程序执行时间
    print("end time is :", endtime,"\n")
    print(endtime - starttime)
