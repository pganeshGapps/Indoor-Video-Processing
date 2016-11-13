# USAGE
# python path_tracking_v1.py --video example.mp4
# python path_tracking_v1.py

# import the necessary packages
from collections import deque
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import cv2
from math import sqrt
import random
from nms import non_max_suppression_fast
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
args = vars(ap.parse_args())

path = {}	# deque storing path of humans
color = {}	# color of path
check = {}	# number of points in path

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

old_circles = []
z = False
firstFrame = None
# keep looping
while True:
	text = "Not Occupied"
	# grab the current frame
	(grabbed, frame) = camera.read();

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	frame = imutils.resize(frame, width=500)

######### BACKGROUND SUBTRACTION
	rects2 = []
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	if firstFrame is None:
		firstFrame = gray
		continue

	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		rects2.append((x,y,x+w,y+h))
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

############ BACKGROUND SUBTRACTION
	# detect people in the frame
	(rects, weights) = hog.detectMultiScale(frame, winStride=(4	, 4), padding=(8, 8), scale=1.05)
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression_fast(rects, overlapThresh=0.65)
	
	# connecting new points to old paths
	if old_circles :
		for (xC,yC) in old_circles :
			flag = False
			for (xA, yA, xB, yB) in pick:
				(A,B) = ((xA+xB)/2, (yA+yB)/2)
				if( sqrt( (A - xC)**2 + (B - yC)**2 ) <= 20) :
					flag = True
					path[(A,B)] = path[(xC,yC)]
					color[(A,B)] = color[(xC,yC)]
					check[(A,B)] = check[(xC,yC)]
					if (A,B) != (xC,yC) :
						del color[(xC,yC)]
						del path[(xC,yC)]
						del check[(xC,yC)]
			if not flag and len(path[xC,yC]) < 10 :
				del path[(xC,yC)]
				del color[(xC,yC)]
				del check[(xC,yC)]

	old_circles = []
    # appending new points to paths
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
		(A,B) = ((xA+xB)/2, (yA+yB)/2)
		if A<frame.shape[0]/2 and B<frame.shape[1]/2 :
			text = "Alert"
		cv2.circle(frame, (A,B), 1, (0, 0, 255), -1)
		old_circles.append((A,B))
		if (A,B) not in path :
			print (str((A,B))+" not in path")
			if not z :
				color[(A,B)] = (0,0,200)
				z = True
			else :
				color[(A,B)] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
			check[(A,B)] = 0
			path[(A,B)] = deque()
		path[(A,B)].appendleft(((xA+xB)/2,yB))
		check[(A,B)] = check[(A,B)] + 1

	text1  = "Not Crowded"
	# crowd detection
	for c1 in old_circles :
		for c2 in old_circles :
			if c1!=c2 and sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 ) <= 50 :
				text1 = "Crowded"

	# printing paths
	for k in path :
		print ("person - "+str(k)+", path size (px) - " + str(len(path[k])) )
		pt = deque()
		if check[k] == 20 :
			pt.append(path[k][0])
			pt.append(path[k][19])
			for i in np.arange(20, len(path[k])):
				pt.append(path[k][i])
			path[k] = pt
			check[k] = 0
		for i in np.arange(1, len(path[k])):
			cv2.line(frame, path[k][i - 1], path[k][i], color[k], 2)

	print("---- next frame ----")
	cv2.rectangle(frame, (0, 0), (frame.shape[0]/2, frame.shape[1]/2), (255, 0, 0), 2)
	cv2.putText(frame, "Status : {}".format(text1), (frame.shape[0]/2+50, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, "Status : {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	cv2.imshow("Human Tracking",frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
