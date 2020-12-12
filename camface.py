# -*- coding: utf-8 -*-
import time
import picamera
import cv2 as cv

fn = 'my_pic.jpg'

# initialize camera
with picamera.PiCamera() as camera:
    # resolution settings
    camera.resolution = (512, 384)
    # preparation taking a photo
    camera.start_preview()
    # Wait a little while preparing
    time.sleep(2)
    # save img
    camera.capture(fn)

    # read img
    img = cv.imread(fn)

    # Convert to mono
    grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # read cascade for face detection
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # read cascade for eye detection
    eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    # Face detection
    facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=2, minSize=(1, 1))

    # Eye detection
    eyerect = eye_cascade.detectMultiScale(grayimg)

    print(facerect)
    print(eyerect)

    # When a face is detected
    if len(eyerect) > 0:
        for rect in eyerect:
            cv.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 255, 0), thickness=3)
        
    # show result img
    cv.imshow('camera', img)
    # write result
    cv.imwrite(fn, img)
    # Wait until any key is pressed
    cv.waitKey(0)
    # close img window
    cv.destroyAllWindows()