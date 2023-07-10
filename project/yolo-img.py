from ultralytics import YOLO
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import cv2
import numpy as np
import imutils
import argparse


img_path = 'folder2/b3.jpg'

#yolo
# model = YOLO('folder2/best.pt')
model = YOLO('folder2/best170-0905.pt')


img = cv2.imread(img_path)
resized_img = cv2.resize(img,(800,700),cv2.INTER_LINEAR)
results = model(resized_img)
# w = 800
# h = 600
# img = cv2.resize(img,(w,h))


boxes = results[0].boxes.numpy()
for box in boxes:
    #conf check
    if box.conf < 0.75:
        continue

    # get tl br 
    x1,y1,x2,y2 = box.xyxy[0].astype(int)
    # print(x1,y1,x2,y2)
    
    #draw box
    cv2.rectangle(resized_img, (x1,y1), (x2,y2), (0,255,255), 3)

    width = x2-x1
    height = y2-y1

    # 20cm = 200px
    ratio = 200/20
    w = width / ratio
    h = height / ratio

    #Text
    cv2.putText(resized_img ,str(w)+'cm',(int(x1+width/2)-50,y1-10),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
    cv2.putText(resized_img ,str(h)+'cm',(x1,int(y1+height/2)),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)

    # cv2.putText(img, str(model.names[int(box.cls)]),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    # cv2.putText(resized_img, str(round(box.conf[0],3)),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    
cv2.imshow('img',resized_img)
cv2.waitKey(0)
cv2.rectangle
cv2.destroyAllWindows()


