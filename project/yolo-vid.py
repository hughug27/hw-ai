from ultralytics import YOLO
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import cv2
import numpy as np
import math


model = YOLO('folder2/best170-0905.pt')

path = 'folder2/vid_Clip2.mp4'

cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
# cap.set(cv2.CAP_PROP_POS_FRAMES,89 * fps)
cv2.namedWindow("vid", cv2.WINDOW_NORMAL)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

out = cv2.VideoWriter('output.mp4', fourcc, fps, (width,height))

while cap.isOpened():
    success, frame = cap.read()

    # stop = cap.get(cv2.CAP_PROP_POS_FRAMES) 
    # print(stop)

    # if stop == 95*fps: 
        # break
    if success:
        results = model(frame)
        boxes = results[0].boxes.numpy()
        for box in boxes:
            if box.conf < 0.79:
                continue

            x1,y1,x2,y2 = box.xyxy[0].astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            width = x2-x1
            height = y2-y1

            # 20cm = 90 px
            ratio = 90/20

            w = math.floor(width / ratio * 10)/10
            h = math.floor(height / ratio * 10)/10

            #width
            cv2.putText(frame ,str(w)+'cm',(int(x1+width/2),y1-10),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
            #height
            cv2.putText(frame ,str(h)+'cm',(x1,int(y1+height/2)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,125),1)
            # conf
            cv2.putText(frame, str(round(box.conf[0],3)),(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)

        out.write(frame)
        # cv2.resizeWindow('vid',900,600)

        cv2.imshow("vid",frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
