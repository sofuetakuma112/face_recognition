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
                loaded_registered_imgs = self.load_registered_img(img_list)
                img_locs = self.detect_human_face(loaded_register_imgs)
                self.loaded_encodings = self.encode_img(loaded_register_imgs, img_locs)
                self.save_registered_encodings(self.loaded_encodings)
        else:
            # register_faceフォルダから画像を読み込みエンコーディングを作成
            print('csvファイルがないのでエンコードし作成します')
            loaded_registered_imgs = self.load_registered_img(img_list)
            img_locs = self.detect_human_face(loaded_register_imgs)
            self.loaded_encodings = self.encode_img(loaded_register_imgs, img_locs)
            self.save_registered_encodings(self.loaded_encodings)


    def load_registered_img(self, img_path):
        loaded_register_imgs = []
        for path in img_path:
            img = fr.load_image_file(path)
            loaded_register_imgs.append(img)
        return loaded_register_imgs


    def detect_human_face(self, face_imgs):
        face_locs = []
        for img in face_imgs:
            print(img.dtype)
            loc = fr.face_locations(img, model="hog")
            face_locs.append(loc)
        return face_locs

    def encode_img(self, face_imgs, face_locs):
        print('エンコード')
        print(face_imgs)
        print(face_locs)
        encodings = []
        for img, loc in zip(face_imgs, face_locs):
            (encoding,) = fr.face_encodings(img, loc)
            encodings.append(encoding)

        print(encodings)
        return encodings
        # self.save_registered_encodings()

    def save_registered_encodings(self, encodings):
        np.savetxt('face_recognition_encodings.csv', encodings)


    def check_face_match(self, captured_encodings):
        print('認証中...')
        # print(self.loaded_encodings)
        # print(captured_encodings)
        # print(self.loaded_encodings.dtype)
        # print(captured_encodings.dtype)
        matches = fr.compare_faces(self.loaded_encodings, captured_encodings)
        print(matches)


    def caputure(self):
        # カメラ初期化
        with picamera.PiCamera() as camera:
            # カメラの画像をリアルタイムで取得するための処理
            with picamera.array.PiRGBArray(camera) as stream:
                camera.resolution = (360, 430)

                while True:
                    # カメラから映像を取得する
                    print('撮影待ち...')
                    camera.capture(stream, 'bgr', use_video_port=True)
                    grayimg = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

                    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                    facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=2, minSize=(100, 100))

                    if (len(facerect) > 0):
                        # cv2.imwrite('face.jpg', stream.array)
                        # img = fr.load_image_file('face.jpg')
                        img_array = []
                        img_array.append(stream.array)
                        # print(type(stream.array))
                        # print(type(img_array))
                        # stream.arrayが画像ファイルなのでこれをエンコードしてmatchesに渡す
                        detected_captured_face = self.detect_human_face(img_array)
                        captured_encodings = self.encode_img(img_array, detected_captured_face)
                        self.check_face_match(captured_encodings)
                        break

                    cv2.imshow('camera', stream.array)

                    stream.seek(0)
                    stream.truncate()

                    if cv2.waitKey(1) > 0:
                        break

                cv2.destroyAllWindows()


if __name__ == '__main__':
    face = Face_recognition()
    face.caputure()  # 撮影待ちになる