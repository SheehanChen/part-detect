# 环境配置
## 环境
```bash
pip install -r requirement.txt
```
### test image & model下载地址
下载后解压缩到当前文件夹下。

[百度云](https://pan.baidu.com/s/1krkLWQZt4C8o1Crgg0S42A)

model也可以直接训练生成。


## 训练

训练的图片已经提取了sift特征并使用PCA处理保存在train_data.txt和train_label.txt文件中了.

run `Part_img_train.py`

## 测试

run `Part_img_test.py`

注程序只会输出识别错误的图片，识别结果在最后输出。

---
# Introduction
本项目对34类零件2400张照片进行了分类识别，零件分为6个大类包括Flat washer、nut、screw_bolt、screw、shoulder screw，共34个细分分类，因为零件在同一个大分类下外形存在较大相似性，我们采用bp神经网络对零件的6个大类进行分类，然后通过图像处理的方法（照片拍摄条件较好，处理比较简单）计算零件的长宽等低级特征对零件的具体型号进行分类。

数据集
数据集是自己制作的，当时还不是很了解主流数据集结构和标注格式，就通过命名来直接对图像的对图像的种类进行标注。下面是数据集的结构，主要包括两个部分，分别是拍照得到的原始图像，和对使用相机标定后的参数对图像进行矫正后的图像。后面我们使用的都是去除相机畸变后的的图像。数据集的制作涉及了相机的标定，拍摄和命名保存等问题。

**文件结构**
<details>
<summary>展开查看</summary>
<pre><code>

├── calibration

│ ├── Flat washer_M22_39_3

│ ├── Flat washer_M24_44_4

│ ├── nut_1_8

│ ├── nut_9_16_12

│ ├── screw_bolt_M12_100

│ ├── screw_bolt_M12_110

│ ├── screw_bolt_M12_120

│ ├── screw_bolt_M12_60

│ ├── screw_bolt_M12_65

│ ├── screw_bolt_M12_70

│ ├── screw_bolt_M12_80

│ ├── screw_bolt_M12_90

│ ├── screw_M5_25

│ ├── screw_M5_30

│ ├── screw_M5_35

│ ├── screw_M5_40

│ ├── screw_M5_45

│ ├── screw_M6_20

│ ├── screw_M6_25

│ ├── screw_M6_30

│ ├── screw_M6_35

│ ├── screw_M6_40

│ ├── screw_M6_45

│ ├── screw_M6_50

│ ├── shoulder screw_M8_20

│ ├── shoulder screw_M8_25

│ ├── shoulder screw_M8_30

│ ├── shoulder screw_M8_35

│ ├── shoulder screw_M8_40

│ ├── shoulder screw_M8_45

│ ├── shoulder screw_M8_50

│ ├── shoulder screw_M8_60

│ ├── shoulder screw_M8_65

│ ├── shoulder screw_M8_75

│ ├── shoulder screw_M8_80

│ ├── shoulder screw_M8_85

│ ├── shoulder screw_M8_90

│ ├── spring washer_M22

│ └── spring washer_M24

└── original

├── Flat washer_M22_39_3

├── Flat washer_M24_44_4
├── nut_1_8

├── nut_9_16_12

├── screw_bolt_M12_100

├── screw_bolt_M12_110

├── screw_bolt_M12_120

├── screw_bolt_M12_60

├── screw_bolt_M12_65

├── screw_bolt_M12_70

├── screw_bolt_M12_80

├── screw_bolt_M12_90

├── screw_M5_25

├── screw_M5_30

├── screw_M5_35

├── screw_M5_40

├── screw_M5_45

├── screw_M6_20

├── screw_M6_25

├── screw_M6_30

├── screw_M6_35

├── screw_M6_40

├── screw_M6_45

├── screw_M6_50

├── shoulder screw_M8_20

├── shoulder screw_M8_25

├── shoulder screw_M8_30

├── shoulder screw_M8_35

├── shoulder screw_M8_40

├── shoulder screw_M8_45

├── shoulder screw_M8_50

├── shoulder screw_M8_60

├── shoulder screw_M8_65

├── shoulder screw_M8_75

├── shoulder screw_M8_80

├── shoulder screw_M8_85

├── shoulder screw_M8_90

├── spring washer_M22

└── spring washer_M24

</code></pre>
</details>

示例图片如下：

![image-20191106201434275](https://user-images.githubusercontent.com/26670635/110235384-aebfd300-7f6a-11eb-9491-0de1f7dae0ee.png)


## 分类模型
根据提供的零件清单购置6大类40种不同型号的零件，使用sift特征+神经网络+尺寸特征进行分类。主要使用sift特征+神经网络分类器对不同种类的零件进行粗分类，输出结果后进入尺寸分类，通过尺寸阈值对不同型号的零件进行细分类。

## SIFT特征
Sift特征提取分为6个环节如图：

![image-20191106202358851](https://user-images.githubusercontent.com/26670635/110235390-b67f7780-7f6a-11eb-8579-bc1995155426.png)


* 构建一个尺度空间

* 怎么找出特征点

* 定位(Difference of Gaussian)极值

* 把对比度小的点剔除

* 关键点方向的确定

* 描述子的构建

* 其中描述子的构建：

![image-20191106202707502](https://user-images.githubusercontent.com/26670635/110235396-bd0def00-7f6a-11eb-8ce4-0d2ec6773253.png)


图4-2的左侧得到了一个关键点的特征方向，关键点周围刻画4*4的方格每个方格的特征方向也是使用梯度值进行计算的，方向360度/8，以45度为间隔每个方格的梯度值大小是不确定的。我们使用方向的大小，也就是向量的长度刻画这个梯度值的大小。（特征点周围由4*4的大方格，每个大方格中有4*4的小方格，由小方格的方向的个数来确定大方格的方向。）然后描述子的大小由4*4*8=128维的向量来描述单位是字节。

## 主成分分析
PCA方法是一种常用的降维方法，因为每张照片提取到的SIFT特征数量不相同，因此描述子向量的个数也不相同。而神经网络的输入向量大小必须相同，因此我们使用PCA方法可以获得相同大小的含有特征信息的数据。输入神经网络进行训练后可以获得比较好的分类效果。

## 神经网络
本程序使用tensorflow框架实现了bp神经网络的训练和测试，作为SIFT特征的分类器，较好得完成了分类任务。

## 具体步骤
首先对摄像头进行标定；

使用标定数据对相机进行正畸；

使用1080p分辨率摄像头在无影棚内对各种零件的各个位置进行拍摄；

制作数据集每种型号的零件50张以上，共2300+张；

对图像进行剪切，只保留零件的最小边缘；

使用脚本随机抽取1%的图片作为测试图像，剩余为训练图像；

对不同型号零件的尺寸进行提取分析尺寸分类的阈值；

对图像灰度化，二值化，提取sift特征；

使用论文中的方法用PCA降维，将特征向量转换为同样的大小；

将计算后的数据保存；

使用TensorFlow框架构建神经网络对图像进行训练（6类）；

使用尺寸对具体型号进行分类（34类）；

使用训练的模型对测试图像进行分类并计算准确率；

多次训练准确率在98%附近，多次超越98%。
