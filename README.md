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

