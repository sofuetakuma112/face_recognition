# -*- coding: utf-8 -*-
import face_recognition as fr
import matplotlib.pyplot as plt
import numpy as np
import picamera
import picamera.array
import cv2
import os
import glob

class Face_recognition:
    def __init__(self):
        # エンコーディングが既にあれば読み込み、無ければ作成する
        if (os.path.exists('face_recognition_encodings.csv')):
            self.loaded_encodings = np.loadtxt('face_recognition_encodings.csv')
            img_list = glob.glob("register_face/*")
            # 既存のエンコーディングの画像枚数とフォルダ内の画像枚数が一致しない場合、再度エンコードする
            if (len(self.loaded_encodings) != len(img_list)):
                print('フォルダに新規画像が追加されたので再度エンコードします')
                # フォルダ内の画像をエンコードする
                self.loaded_encodings = self.load_registered_img(img_list)
                self.save_registered_encodings(self.loaded_encodings)
        else:
            # register_faceフォルダから画像を読み込みエンコーディングを作成
            print('csvファイルがないのでエンコードし作成します')
            self.loaded_encodings = self.load_registered_img(img_list)
            self.save_registered_encodings(self.loaded_encodings)


    def load_registered_img(self, img_path):
        loaded_register_imgs = []
        for path in img_path:
            img = fr.load_image_file(path)
            loaded_register_imgs.append(img)
        self.detect_human_face(loaded_register_imgs)


    def detect_human_face(self, face_imgs):
        face_locs = []
        for img in face_imgs:
            loc = fr.face_locations(img, model="hog")
            face_locs.append(loc)
        self.encode_img(face_imgs, face_locs)

    def encode_img(self, face_imgs, face_locs):
        # 戻り値をencodingsにすると汎用性を持つ
        encodings = []
        for img, loc in zip(face_imgs, face_locs):
            (encoding,) = fr.face_encodings(img, loc)
            encodings.append(encoding)
        return encodings
        # self.save_registered_encodings()

    def save_registered_encodings(self, encodings):
        np.savetxt('face_recognition_encodings.csv', encodings)


    def check_face_match(self, captured_encodings):
        matches = fr.compare_faces(self.loaded_encodings, captured_encodings)
        print(matches)


    def caputure(self):
        # カメラ初期化
        with picamera.PiCamera() as camera:
            # カメラの画像をリアルタイムで取得するための処理
            with picamera.array.PiRGBArray(camera) as stream:
                camera.resolution = (512, 384)

                while True:
                    # カメラから映像を取得する
                    print('撮影待ち...')
                    camera.capture(stream, 'bgr', use_video_port=True)
                    grayimg = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

                    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                    facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=2, minSize=(100, 100))

                    if (len(facerect) > 0):
                        cv2.imwrite('face.jpg', stream.array)
                        # stream.arrayが画像ファイルなのでこれをエンコードしてmatchesに渡す
                        captured_encodings = self.detect_human_face([stream.array])
                        self.check_face_match(captured_encodings)
                        break

                    # cv2.imshow('camera', stream.array)

                    # stream.seek(0)
                    # stream.truncate()

                    if cv2.waitKey(1) > 0:
                        break

                cv2.destroyAllWindows()


if __name__ == '__main__':
    face = Face_recognition()
    face.caputure()  # 撮影待ちになる