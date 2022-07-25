import cv2
import random

names = []
with open('class.txt', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')
# print(class_name)
net = cv2.dnn.readNet('yolov4-tiny_last.weights', 'yolov4-tiny.cfg')
#net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
img = cv2.imread('image.jpg')
classes, confidences, boxes = model.detect(img, confThreshold=0.45, nmsThreshold=0.45)
#if classes=="[0]"
for (classId, confidence, box) in zip(classes, confidences, boxes):
    label = '%s: %.2f' % (names[classId[0]], confidence)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)
    left, top, width, height = box
    top = max(top, labelSize[1])
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    cv2.rectangle(img, box, color=(b, g, r), thickness=4)
    cv2.rectangle(img, (left - 1, top - labelSize[1]), (left + labelSize[0], top), (b, g, r), cv2.FILLED)
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255 - b, 255 - g, 255 - r), 2)
cv2.namedWindow('detect out', cv2.WINDOW_NORMAL)
cv2.imshow('detect out', img)

cv2.waitKey(0)
