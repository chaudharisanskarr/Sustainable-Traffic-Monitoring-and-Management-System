import cv2
import numpy as np


#web cam or video

cap = cv2.VideoCapture('video.mp4')

import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('video.mp4')

# Define parameters
min_width_rect = 80
min_height_rect = 80
count_line_position = 650
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
detect = []
offset = 6
counter = 0

# Set the desired frame width and height
frame_width = 1280
frame_height = 720

while True:
    ret, frame1 = cap.read()
    if not ret:
        break  # Break the loop when the video ends

    # Resize the frame to the desired resolution
    frame1 = cv2.resize(frame1, (frame_width, frame_height))

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtractor
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame1, (20, count_line_position), (frame_width - 20, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = (int(x + w / 2), int(y + h / 2))
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        # Count vehicles
        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
            cv2.line(frame1, (25, count_line_position), (frame_width - 25, count_line_position), (255, 127, 0), 3)
            detect.remove((x, y))
            print("Vehicle Counter: " + str(counter))

    cv2.putText(frame1, "Vehicle Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()

min_width_rect = 80
min_hieght_rect = 80

count_line_position = 550

# Substactor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6
counter = 0


while True:
    ret,frame1 = cap.read()
    
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    
    # apply
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #line position and length
    cv2.line(frame1,(10,count_line_position),(1300,count_line_position),(255,127,0),3)
    
    
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rect) and (h>= min_hieght_rect)
        if not validate_counter:
            continue
        
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255), -1)
        
        #count
        
        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter += 1
            cv2.line(frame1,(25,count_line_position),(600,count_line_position),(255,127,0),3)
            detect.remove((x,y))
            print("Vehicle Counter: "+str(counter))
            
    cv2.putText(frame1,"Vehicle Counter : "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
   
   
    #cv2.imshow('Detector', dilatada)
    cv2.imshow('Video Original', frame1)
    
    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()
cap.release()