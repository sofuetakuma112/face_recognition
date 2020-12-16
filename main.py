# -*- coding: utf-8 -*-
import face_recognition as fr
import matplotlib.pyplot as plt
import numpy as np
import picamera
import picamera.array
import cv2
import os
import glob
import pickle
import serial
import time

class Face_recognition:
    def __init__(self):
        self.max_pixel = 320000
        self.resize_width = 480
        self.captured_encodings = None
        self.matches = None
        self.img_list = glob.glob("register_face/*")
        old_img_list = None
        # エンコーディングが既にあれば読み込み、無ければ作成する
        if (os.path.exists('face_recognition_encodings.csv')):  # 二回目以降
            print('エンコーディング読み込み')
            self.loaded_encodings = np.loadtxt('face_recognition_encodings.csv')
            read_file = open('register_face_path_list.txt', 'rb')
            old_img_list = pickle.load(read_file)
            # 現在のファイル名と前回のエンコード時のファイル名が一致しないなら再度エンコード
            if (old_img_list != self.img_list):
                print('フォルダに新規画像が追加されたので再度エンコードします')
                write_file = open('register_face_path_list.txt', 'wb')
                pickle.dump(self.img_list, write_file)
                self.resize()  # 解像度が大きい画像をリサイズする
                self.encode()  # フォルダ内の画像をエンコードする
        else:  # 初回
            # register_faceフォルダから画像を読み込みエンコーディングを作成
            print('csvファイルが見つからないので作成します')
            write_file = open('register_face_path_list.txt', 'wb')
            pickle.dump(self.img_list, write_file)
            self.resize()
            self.encode()


    def load_registered_img(self):
        """
        register_faceフォルダ内のファイルを読み込む
        """
        loaded_register_imgs = []
        for path in self.img_list:
            img = fr.load_image_file(path)
            loaded_register_imgs.append(img)
        return loaded_register_imgs


    def detect_human_face(self, face_imgs):
        """
        画像から顔部分を検出する
        """
        face_locs = []
        for img in face_imgs:
            loc = fr.face_locations(img, model="hog")  # 顔が検出できない画像は loc = [] になる
            if (len(loc) != 0):
                face_locs.append(loc)
            else:
                print('顔が検出できませんでした')
        return face_locs

    def encode_img(self, face_imgs, face_locs):
        """
        画像と検出した顔部分のデータから特徴量を求める
        """
        print('エンコード')
        encodings = []
        for img, loc in zip(face_imgs, face_locs):
            (encoding,) = fr.face_encodings(img, loc)
            encodings.append(encoding)
        return encodings

    def encode(self):
        loaded_registered_imgs = self.load_registered_img()
        img_locs = self.detect_human_face(loaded_registered_imgs)
        self.loaded_encodings = self.encode_img(loaded_registered_imgs, img_locs)  # [array([]), array([]), array([]), ...]
        self.save_registered_encodings(self.loaded_encodings)
        self.loaded_encodings = np.loadtxt('face_recognition_encodings.csv')  # [[], [], [], ...]

    def save_registered_encodings(self, encodings):
        """
        register_face内の特徴量のデータを保存する
        """
        np.savetxt('face_recognition_encodings.csv', encodings)


    def check_face_match(self):
        print('認証')
        self.matches = fr.compare_faces(self.loaded_encodings, self.captured_encodings)


    def caputure(self):
        """
        PiCameraを起動して人の顔を検知したらシャッターを切る
        """
        # カメラ初期化
        while True:
            with picamera.PiCamera() as camera:
                # カメラの画像をリアルタイムで取得するための処理
                with picamera.array.PiRGBArray(camera) as stream:
                    camera.resolution = (640, 480)

                    # カメラから映像を取得する
                    print('撮影待ち...')
                    camera.capture(stream, 'bgr', use_video_port=True)
                    grayimg = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

                    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                    facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=2, minSize=(100, 100))

                    if (len(facerect) > 0):
                        img_array = []
                        img_array.append(stream.array)
                        # stream.arrayが画像ファイルなのでこれをエンコードしてmatchesに渡す
                        detected_captured_face = self.detect_human_face(img_array)  # [[(187, 282, 295, 175)]]
                        try:
                            self.captured_encodings = self.encode_img(img_array, detected_captured_face)
                        except:
                            print('顔検出失敗')
                            cv2.destroyAllWindows()
                            continue
                        break

                    # cv2.imshow('camera', stream.array)

                    stream.seek(0)
                    stream.truncate()

                    if cv2.waitKey(1) > 0:
                        break
        cv2.destroyAllWindows()


    def resize(self):
        """
        ピクセルサイズが大きい画像をリサイズする
        """
        big_size_img_path_list = self.check_img_size()  # 解像度が大きい画像のみのpathの配列
        resized_imgs = []
        if (len(big_size_img_path_list) != 0):
            print('解像度が大きい画像があるのでリサイズします')
            for big_img_path in big_size_img_path_list:  # リサイズしてpathとimgの辞書にする
                resized_img = self.resize_img(big_img_path, self.resize_width)
                resized_imgs.append({'path': big_img_path, 'img': resized_img})
            self.overwrite_img(resized_imgs)  # 辞書を元に画像を上書き
        else:
            print('リサイズが必要な画像はありませんでした')


    def check_img_size(self):
        """
        解像度が大きい画像のpathを配列で返す
        """
        big_size_img_path = []
        for path in self.img_list:
            img = cv2.imread(path)
            h, w = img.shape[:2]
            if (h * w >= self.max_pixel):
                big_size_img_path.append(path)

        return big_size_img_path



    def resize_img(self,img_path, width):
        """
        指定したwidthでアスペクト比を維持しながらリサイズ
        """
        img = cv2.imread(img_path)
        h, w = img.shape[:2]  # 画像の高さと幅をそれぞれ分割代入
        height = round(h * (width / w))
        dst = cv2.resize(img, dsize=(width, height))
        return dst


    def overwrite_img(self, resized_img):
        """
        prams: 辞書型
        渡された画像の配列を元の画像に上書き保存する
        """
        for i in resized_img:
            path = i['path']
            img = i['img']
            cv2.imwrite(path, img)


    def pass_result_to_arduino(self):
        """
        Arduinoに結果を渡す
        """
        if (True in self.matches):
            print('OK')
            with serial.Serial('/dev/ttyACM0', 9600, timeout=0.01) as ser:
                ser.write('1'.encode('utf-8'))
        else:
            print('NO')
            with serial.Serial('/dev/ttyACM0', 9600, timeout=0.01) as ser:
                ser.write('0'.encode('utf-8'))


if __name__ == '__main__':
    face = Face_recognition()
    count = 0
    while True:
        face.caputure()  # 撮影待ちになる
        try:
            face.check_face_match()
        except:
            print('撮影した写真から顔が検出できませんでした')
            continue
        face.pass_result_to_arduino()  # 結果表示
        time.sleep(5)
    print('シャットダウンします')