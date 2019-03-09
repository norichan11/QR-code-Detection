# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
from urllib.request import urlopen, Request
from face_detector2 import face_detector2

def decode(im):
    # Find barcodes or QR codes
    decodedObjects = pyzbar.decode(im)
    # Print results
    #for obj in decodedObjects:
    #    print('Type : ', obj.type)
    #    print('Data : ', obj.data, '\n')
    return decodedObjects
    
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)
font = cv2.FONT_HERSHEY_SIMPLEX

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    decodedObjects = decode(im)

    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points;

        # Number of points in the convex hull
        n = len(hull)
        # Draw the convext hull
        for j in range(0, n):
            cv2.line(image, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

        x = decodedObject.rect.left
        y = decodedObject.rect.top

        print(x, y)
        print('Type : ', decodedObject.type)
        #print('Data : ', decodedObject.data.decode('UTF-8'), '\n')
        barCode = decodedObject.data.decode('UTF-8')
        cv2.putText(image, barCode, (x, y), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        print('UserId:', barCode)

        if len(barCode) > 10:
            print("Face detecting...")

            # Using OpenCV and the trained yml-file:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('face_detector/trainer_' + barCode + '/trainer.yml')

            # Using xml:
            cascade_path = "face_detector/haarcascades/haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)

            img_id=0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(80, 80))

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                img_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                print('img_id=', img_id)
                print('confidence=', conf)

                if (conf < 50):
                    username = "Pass: " + barCode
                    confidence = "  %s" % (round(100 - conf))
                    r = Request("https://0da3c1a2.ngrok.io/welcom/" + barCode)
                    responce = urlopen(r)
                    #str1 = "Welcome, " + str(username)
                    cv2.putText(im, "Welcome, " , (x, y + h), font, 0.55, (0, 255, 0), 1)
                else:
                    username = "Unknown"
                    confidence = "  %s" % (round(100 - conf))
                    r = Request("https://0da3c1a2.ngrok.io/tryagain/" + barCode)
                    responce = urlopen(r)
                    cv2.putText(im, "Please Try Again", (x, y + h), font, 0.55, (0, 255, 0), 1)

                # cv2.cv.PutText(cv2.cv.fromarray(im), str(Id), (x, y   h), font, 255)
                cv2.putText(im, str(username), (x, y+h), font, 0.55, (0, 255, 0), 1)

    

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        
cv2.destroyAllWindows()
