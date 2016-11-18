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
import time
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
args = vars(ap.parse_args())

path = {}	# deque storing path of humans
color = {}	# color of path
check = {}	# number of points in path

ticks = time.time()
starttime = 18
timelimit = 30

firstFrame = None
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
firstRectangle = 0
counter = 10
disp = False
text4 = "Entry"
# keep looping
while True:
	if ticks+starttime + timelimit - time.time() < 15 :
		text4 = "Exit"
	if disp == False :
		counter = 10
	text = "Not Occupied"
	# grab the current frame
	(grabbed, frame) = camera.read();

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	frame = imutils.resize(frame, width=500)

######### BACKGROUND SUBTRACTION
	rects1 = []
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	if firstFrame is None:
		firstFrame = gray
		continue

	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	included = {}
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		rects1.append((x,y,x+w,y+h))
		included[(x,y,x+w,y+h)] = False
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

	pick1 = []
	#cv2.rectangle(frame,(150,100),(200,150),(0,0,0),2)
	#combining blobs
	for (x1,y1,x2,y2) in rects1 :
		(a,b) = ((x1+x2)/2,(y1+y2)/2) # center1
		if included[(x1,y1,x2,y2)] == False :
			included[(x1,y1,x2,y2)] = True
			arr = []
			arr.append((x1,y1,x2,y2))
			for (x3,y3,x4,y4) in rects1 :
				(c,d) = ((x3+x4)/2,(y3+y4)/2) #center2
				if sqrt((a-c)*(a-c)+(b-d)*(b-d)) < 100 and included[(x3,y3,x4,y4)] == False :
					arr.append((x3,y3,x4,y4))
					included[(x3,y3,x4,y4)] = True
			#drawing it
			minx = 10000000
			miny = 10000000
			maxx = 0
			maxy = 0
			for (x1,y1,x2,y2) in arr :
				if minx!= None and minx > x1 :
					minx = x1
				if miny!= None and miny > y1 :
					miny = y1
				if maxx!= None and maxx < x2 :
					maxx = x2
				if maxy!= None and maxy < y2 :
					maxy = y2
			(A,B) = ((minx+miny)/2,(maxx+maxy)/2)
			#cv2.line(frame,(100,140),(200,140),(100,100,200),2)
			(C,D) = ((A+B)/2,(A+5*B)/6)
			print ((C,D))
			#print "sdfjjsdfjads - hi" + str((C,D))
			if C >=130 and C <= 145 and D >= 100 and D <=160 :
				#cv2.putText(frame, "{}".format(text1), (frame.shape[0]/2+50, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				disp = True
				#cv2.putText(frame, "{}".format("Entry"), (frame.shape[0]/2+100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
			else :
				disp = False
			cv2.rectangle(frame, (minx, miny), (maxx, maxy), (255, 0, 255), 2)
			if A <=180 :
				pick1.append((minx,miny,maxx,maxy))
			else :
				text = "Occupied"
############ BACKGROUND SUBTRACTION
	# detect people in the frame
	'''(rects, weights) = hog.detectMultiScale(frame, winStride=(4	, 4), padding=(8, 8), scale=1.02)
	print weights
	tt = []
	for i in range(0,len(weights)) :
		if weights[i] > 1 :
			(x, y, w, h) = rects[i]
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			tt.append([x,y,x+w,y+h])
	'''
	#rects = np.array(tt)
	rects = np.array(pick1)#[[x, y, x + w, y + h] for (x, y, w, h) in rects])

	pick = non_max_suppression_fast(rects, overlapThresh=0.65)
	
	# connecting new points to old paths
	if old_circles :
		for (xC,yC) in old_circles :
			flag = False
			for (xA, yA, xB, yB) in pick:
				(A,B) = ((xA+xB)/2, (yA+yB)/2)
				if( sqrt( (A - xC)**2 + (B - yC)**2 ) <= 20) :
					flag = True
					if (xC,yC) in path :
						path[(A,B)] = path[(xC,yC)]
						color[(A,B)] = color[(xC,yC)]
						check[(A,B)] = check[(xC,yC)]
						if (A,B) != (xC,yC) :
							del color[(xC,yC)]
							del path[(xC,yC)]
							del check[(xC,yC)]
			if not flag and len(path[xC,yC]) < 10 :
				if (xC,yC) in path :
					del path[(xC,yC)]
					del color[(xC,yC)]
					del check[(xC,yC)]

	old_circles = []
    # appending new points to paths
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 255), 2)
		(A,B) = ((xA+xB)/2, (yA+yB)/2)
		if A >= 350 and A<=frame.shape[0]+218 and B>=30 and B<=frame.shape[1]/2 + 20:
			text = "Occupied"
		cv2.circle(frame, (A,B), 1, (0, 0, 255), -1)
		old_circles.append((A,B))
		if (A,B) not in path :
			print (str((A,B))+" not in path")
			if not z :
				color[(A,B)] = (0,255,255)
				z = True
			else :
				color[(A,B)] = (0,255,255) #(random.randint(0,255),random.randint(0,255),random.randint(0,255))
			check[(A,B)] = 0
			path[(A,B)] = deque()
		path[(A,B)].appendleft(((xA+xB)/2,(yA+5*yB)/6))
		check[(A,B)] = check[(A,B)] + 1

	text1  = "Not Crowded"
	flag = False
	# crowd detection
	for c1 in old_circles :
		for c2 in old_circles :
			if c1!=c2 and sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 ) <= 100 :
				text1 = "Crowded"
				flag = True
				break
		if flag :
			break
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

	cv2.rectangle(frame,(0,0),(500,20),(0,0,0),19)
	cv2.line(frame,(300,0),(300,90),(255,0,0),5)
	cv2.line(frame,(300,90),(frame.shape[0]+218, 130),(255,0,0),5)
	#cv2.rectangle(frame, (350, 30), (frame.shape[0]+218, frame.shape[1]/2+20), (255, 0, 0), 2)
	if time.time() > ticks + starttime and time.time() < ticks + starttime + timelimit :
		text3 = str(round(ticks+starttime + timelimit - time.time(),2))
		cv2.putText(frame, "{}".format("ALERT - "+text3), (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		#os.system('say "beep"')
	#cv2.putText(frame, "{}".format(text1), (frame.shape[0]/2+50, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	if disp :
		cv2.putText(frame, "{}".format(text4), (frame.shape[0]/2+100, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, "{}".format(text), (frame.shape[0]/2+250, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),	(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Human Tracking",frame)
	
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
