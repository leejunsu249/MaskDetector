import numpy as np
from datetime import datetime 
import pytz 
import cv2
from imutils.video import FPS
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1)
ap.add_argument("-c", "--confidence", type=float, default=0.45)
ap.add_argument("-t", "--threshold", type=float, default=0.3)
ap.add_argument("-u", "--use-gpu", type=bool, default=0)
args = vars(ap.parse_args())


LABELS = ["NoMask","Mask"]

COLORS = [[0,0,237],[241,95,95]]

weightsPath = ""
configPath = ""


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if args["use_gpu"]:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

W = None
H = None

video = cv2.VideoCapture(0)
writer = None
fps = FPS().start()

while True:

    (grabbed, frame) = video.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (864, 864),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])
    
    border_size=100
    border_text_color=[255,255,255]
    frame = cv2.copyMakeBorder(frame, border_size,0,0,0, cv2.BORDER_CONSTANT)
    
    filtered_classids=np.take(classIDs,idxs)
    mask_count=(filtered_classids==1).sum()
    nomask_count=(filtered_classids==0).sum()

    text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
    cv2.putText(frame,text, (0, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,border_text_color, 2)

    ratio=nomask_count/(mask_count+nomask_count+0.000001)

    
    if ratio>=0.1 and nomask_count>=3:
        text = "Danger !"
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[26,13,247], 2)
        
    elif ratio!=0 and np.isnan(ratio)!=True:
        text = "Caution !"
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,255], 2)

    else:
        text = "Safe "
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,0], 2)
    if len(idxs) > 0:

        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = LABELS[classIDs[i]]
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.6, color, 2)

    if args["display"] > 0:
      
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF
       
        if key == ord("q"):
            break

    fps.update()

fps.stop()
