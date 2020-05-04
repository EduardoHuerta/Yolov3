from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
from tensorflow.keras import datasets, layers, models
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import tensorflow as tf
import YOLO_App

cap = cv2.VideoCapture('John.mp4')
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

Size=np.shape(img)
model1 = tf.keras.models.load_model('yolov3_5Clases.h5',compile=False)
#labels = ["person", "car", "cat", "dog","cell phone"]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
input_w, input_h = 416, 416
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
class_threshold = 0.5
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (228,200,50)
lineType               = 2

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
#    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image, image_w, image_h = YOLO_App.preprocess_image(frame, (input_w, input_h))
    yhat = model1.predict(image)
    boxes = list()
    for i in range(len(yhat)):
	# Decodificar yhat para obtener bounding boxes
    # INPUT = yhat un array a la vez, anchors un array a la vez, umbral deteccion y dimensiones de la imagen
    	boxes += YOLO_App.decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
    YOLO_App.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
    YOLO_App.do_nms(boxes, 0.5)
    v_boxes, v_labels, v_scores = YOLO_App.get_boxes(boxes, labels, class_threshold)

    # draw what we found

    frame=YOLO_App.draw_boxes_video(frame, v_boxes, v_labels, v_scores)
#    
#    img_out = LBP(img, 3)
    # Display the resulting frame
    cv2.imshow('frame_YOLO',frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()