# -*- coding: utf-8 -*-
import picamera
import picamera.array
import cv2

# カメラ初期化
with picamera.PiCamera() as camera:
    # カメラの画像をリアルタイムで取得するための処理
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (512, 384)

        while True:
            # カメラから映像を取得する
            camera.capture(stream, 'bgr', use_video_port=True)
            grayimg = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=2, minSize=(100, 100))

            if (len(facerect) > 0):
                cv2.imwrite('my_pic2.jpg', stream.array)
                # face recognition process
                break

            cv2.imshow('camera', stream.array)

            stream.seek(0)
            stream.truncate()

            if cv2.waitKey(1) > 0:
                break

        cv2.destroyAllWindows()