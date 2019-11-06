''' 该程序用于从图片集中随机得抽取图片作为测试数据集 '''
import os, random, shutil


def moveFile(fileDir, tarDir):  # 移动

    pathDir = os.listdir(fileDir)    # 取图片的原始路径
    filenumber=len(pathDir)
    rate=0.1    # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=int(filenumber*rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.move(fileDir+name, tarDir+name)
    return


def Tra_folder(dir):

    # 遍历文件夹中子文件夹
    if os.path.exists(dir):
        path_dir = os.path.abspath(dir)
        # path_dir=path_dir
        for i in os.listdir(path_dir):
            path_i = os.path.join(path_dir, i)
            path_i=path_i+'\\'
            nfolder_path=os.path.join(nfileDir, i)
            nfolder_path=nfolder_path+'\\'
            os.makedirs(nfolder_path)
            moveFile(path_i,nfolder_path)


if __name__ == '__main__':
    fileDir = "cut_image_merge\\"    # 源图片文件夹路径
    nfileDir = "test image_2\\"    # 移动到新的文件夹路径
    Tra_folder(fileDir)
