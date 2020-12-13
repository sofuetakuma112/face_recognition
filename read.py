import numpy as np
import os
import glob

# カスケード分類器が既にあれば読み込み、無ければ作成する
if (os.path.exists('face_recognition_cascade.csv')):
    loaded_cascade = np.loadtxt('face_recognition_cascade.csv')
    imgList = glob.glob("register_face/*")
    print(imgList)
    print(len(imgList))
    # 撮影した写真の人物と一致するか照合
# else:
    # register_faceフォルダから画像を読み込みカスケード分類器を作成


# loaded_cascade = np.loadtxt('face_recognition_cascade.csv')
# np.loadtxt('aaa')
# print(loaded_cascade)
# print(loaded_cascade.shape)
# print(len(loaded_cascade))