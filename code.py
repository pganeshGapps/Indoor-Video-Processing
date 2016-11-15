import cv2
import numpy as np
from copy import deepcopy
import time
from sklearn.cluster import AffinityPropagation

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):	
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

cap = cv2.VideoCapture('video3_.avi')
count = 0 #counts the no of frames read till now
nframe = 100 #no of frames needed to initialize the background
cols = 160
rows = 160
flag = 0
move = 0
avg = np.zeros([160,160],dtype=np.uint8)
avg_temp = np.zeros([160,160],np.uint)
sumq = np.zeros([160,160],np.uint)
cur_back = np.zeros([160,160],dtype=np.uint8)
buf_back = np.zeros([160,160],dtype=np.uint8)
colq = Queue()

#to form clusters
count_5=0
arr=np.zeros(shape=(0, 2), dtype=np.uint8)
cur_cent=last_cent=[0,0]
def dist(cur_cent, last_cent):
	dis=(cur_cent[0]-last_cent[0])**2 + (cur_cent[1] - last_cent[1])**2
	return dis
cluster_centres_q = np.zeros(shape=(0,2),dtype=np.int64)

while(cap.isOpened()):
	#time.sleep(0.1)
	count = count + 1
	#print(count)
	#print(nframe)
	ret, pure_img = cap.read()
	img = cv2.resize(pure_img,(160,160))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	colq.enqueue(gray)
	if(count < nframe):
		#avg[:]=0 #no need as avg is initialized to be zero matrix
		sumq = sumq + gray 
	
	else:
		temp = colq.dequeue()
		sumq = sumq + gray - temp
		avg_temp = sumq/nframe
	
	high_value_indices = avg_temp>255
	avg_temp[high_value_indices] = 255
	avg=avg_temp
	avg=avg.astype(np.uint8)	

#	print(gray)
#	print(sumq)
#	print(avg)
#	
#	print(avg.shape)
	#print(avg_show.shape)


	cur_back = avg
	if(flag == 0):
		#buf_back[:] = 0 #no need as buf_back is initialized to be zero matrix
		flag = 10
	if(flag == 10 and count >= nframe):
		buf_back = cur_back
		flag = 20
	
	sub = cv2.absdiff(cur_back,buf_back)

	img_show = cv2.resize(img,(400,400))
	#img_show = img_show.astype(int)
	#print(img_show.shape)
	#print(img_show)
	#time.sleep(1)
	cv2.imshow("img",img_show)

	gray_show = cv2.resize(gray,(400,400))
	#gray_show = gray_show.astype(int)
#	cv2.imshow("gray",gray_show)

	#print(cur_back)
	cur_back_show = cv2.resize(cur_back,(400,400))
	#cur_back_show = cur_back_show.astype(int)
	#print(cur_back_show)
#	cv2.imshow("cur_back_show",cur_back_show)
	
	buf_back_show = cv2.resize(buf_back,(400,400))
	#buf_back_show = buf_back_show.astype(int)
#	cv2.imshow("buf_back_show",buf_back_show)

	sub = cv2.resize(sub,(400,400))
	#sub=sub.astype(int)
	cv2.imshow("Abandoned Objects",sub)

	ret_s,sub_t = cv2.threshold(sub,50,255,0)
	mask = np.zeros(gray.shape,np.uint8)

	contours, hier = cv2.findContours(sub_t,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	#im2, contours, hier = cv2.findContours(sub_t,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	count_5 += 1
	for cnt in contours:
    		if 200<cv2.contourArea(cnt)<5000:
        		#cv2.drawContours(sub,[cnt],0,(0,255,0),2)
	       		cv2.drawContours(mask,[cnt],0,255,-1)
			M = cv2.moments(cnt)
			cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
			#cv2.circle(sub,(cx,cy),10,255,-1)
			if(count_5<=5):
				arr = np.append(arr, [[cx,cy]], axis=0)

	print(arr)
	print(count_5)
	
	if(count_5==5):
		count_5 = 0
		print len(arr)
		if (len(arr) == 0):
			pass 
		else:
			affin = AffinityPropagation()
			affin.fit_predict(arr)
			centroids = affin.cluster_centers_	
			labels = affin.labels_
			max_label_index = labels.argmax()
			biggest_clust_ind = labels[max_label_index]
			print(labels)
			print(centroids)
			print("len_labels",len(labels),"biggest cluster's index", biggest_clust_ind, "len_centroids", len(centroids))
			print("type of biggest_clust_ind", type(biggest_clust_ind))
			if( type(biggest_clust_ind)==np.int64):
				biggest_clust_cent = centroids[biggest_clust_ind]
				cx = np.uint8(biggest_clust_cent[0])
				cy = np.uint8(biggest_clust_cent[1])
				cv2.rectangle(sub,(cx-15,cy-15),(cx+15,cy+15),(255,255,255),1)
				cv2.rectangle(img,(cx-7,cy-7),(cx+7,cy+7),(0,255,0),2)
				#cv2.drawContours(sub, contours, -1, (0,255,0), 3)
				#finallly reinitializing the np_array arr
				last_cent = cur_cent
				cur_cent = [cx,cy]
				cluster_centres_q = np.append(cluster_centres_q, [cur_cent], axis=0)
			else:
				pass
			arr=np.zeros(shape=(0, 2), dtype=np.uint8)
	dista=dist(cur_cent, last_cent)	
	print("distance b/w centroid of last & current frame",dista)
	#if(0 < dista < 25 ):
	if(cur_cent==last_cent and last_cent!=[0,0]):
		cv2.rectangle(sub,(cx-15,cy-15),(cx+15,cy+15),(255,255,255),1)
		cv2.rectangle(img,(cx-7,cy-7),(cx+7,cy+7),(0,255,0),2)
		
	if(len(cluster_centres_q)==6):
		temp_a = deepcopy(cluster_centres_q[0])
		temp_b = cluster_centres_q[-1]
		if(temp_a[0] == temp_b[0] and temp_a[1] == temp_b[1]):
			print ("warning, abandoned object detected")
			#font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) #Creates a font
			font = cv2.FONT_HERSHEY_SIMPLEX
			text_x = cx-10 #position of text
			text_y = cy-20 #position of text
			cv2.putText(sub,"Warning", (text_x,text_y),font,1, (255,255,255)) #Draw the text
		cluster_centres_q = cluster_centres_q[1:]


	avg_show = cv2.resize(avg,(400,400))
#	cv2.imshow("avg",avg_show)
	cv2.imshow("Abandoned Objects",sub)
	if(move==0):
		move=1
		cv2.moveWindow("gray", 400,20)
		cv2.moveWindow("img", 0,20)
		cv2.moveWindow("cur_back_show", 800,20)
		cv2.moveWindow("buf_back_show", 400,420)
		cv2.moveWindow("Abandoned Objects", 20,220)
		cv2.moveWindow("avg", 800,420)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()	
