import cv2
import sys
import argparse
import imutils
import datetime
import time
# Get user supplied values

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(args["video"])
text = "Uncrowded"
ticks = time.time()
starttime = 20
timelimit = 20
inside = False
counter = 50
while True :
    (grabbed,frame) = camera.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    '''
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    '''
    print str(time.time())
    if time.time() > ticks + starttime and time.time() < ticks + starttime + timelimit :
        text = "Crowded"
        inside = True
    else :
        text = "Uncrowded"

    if (time.time() - ticks + starttime + timelimit) > 90 :
        inside = False
        text = "Uncrowded"
    cv2.putText(frame, "{}".format(text), (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print("Found {0} faces!".format(len(faces)))
    #cv2.putText(frame, "{}".format("Uncrowded"), (frame.shape[0]/2+250, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Faces found", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()