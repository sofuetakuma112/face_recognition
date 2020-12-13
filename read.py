# -*- coding: utf-8 -*-
import face_recognition as fr
import matplotlib.pyplot as plt
import numpy as np
import picamera
import picamera.array
import cv2
import os
import glob

# カスケード分類器が既にあれば読み込み、無ければ作成する
if (os.path.exists('face_recognition_encodings.csv')):
    img_list = glob.glob("test/*")
    loaded_register_imgs = []
    for path in img_list:
        img = fr.load_image_file(path)
        loaded_register_imgs.append(img)

    face_locs = []
    print(type(loaded_register_imgs))  # 読み込んだ画像を配列に格納
    print(loaded_register_imgs)
    for img in loaded_register_imgs:
        print(type(img))
        print(img.dtype)
        print(img)
        print(img.shape)
        loc = fr.face_locations(img, model="hog")
        face_locs.append(loc)
    # 撮影した写真の人物と一致するか照合
# else:
    # register_faceフォルダから画像を読み込みカスケード分類器を作成


# loaded_cascade = np.loadtxt('face_recognition_cascade.csv')
# np.loadtxt('aaa')
# print(loaded_cascade)
# print(loaded_cascade.shape)
# print(len(loaded_cascade))