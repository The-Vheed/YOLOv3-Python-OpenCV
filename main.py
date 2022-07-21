#############################################
# Object detection - YOLO v3 - OpenCV
# Modificaion Author: David Mbatuegwu
# Twitter: @The-Vheed
############################################

# Resource reference: Arun Ponnusamy (July 16, 2018) http://www.arunponnusamy.com

import cv2
import numpy as np
import requests
import os
import wget

'''You can set the video stream source to an online or offline .mp4 video,
a camera feed, or even an online video (updated images) source'''

# source = 0
# source = 'https://vod-progressive.akamaized.net/exp=1658272921~acl=%2Fvimeo-prod-skyfire-std-us%2F01%2F4746%2F18%2F473734229%2F2111812190.mp4~hmac=270dd65e0deec3c558988abcbd30c7938e19e95010ee19b153f761cb232ed80c/vimeo-prod-skyfire-std-us/01/4746/18/473734229/2111812190.mp4?download=1&filename=pexels-pat-whelen-5737543.mp4'
source = 'sample_input.mp4'  ############################################################################################
skip_factor = 2

if not 'yolov3.weights' in os.listdir('./'):
    wget.download('https://pjreddie.com/media/files/yolov3.weights', 'yolov3.weights')

image = 'streetview.jpg'
config = 'yolov3.cfg'
weights = 'yolov3.weights'
classes = 'yolov3.txt'


def fetchframe(source):
    if type(source) == int or (type(source) == str and (source[-4:] == '.mp4')):
        ret, frame = cap.read()
    else:
        img_resp = requests.get(source)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        ret = True
    return ret, frame


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    global image
    label = str(classes[class_id]).capitalize() + ': ' + str(round(confidence * 100, 1)) + '%'

    color = COLORS[class_id]

    if classes[class_id] == 'person':
        color = np.array([255.0, 0.0, 0.0])
    elif classes[class_id] == 'motorcycle':
        color = np.array([0.0, 0.0, 255.0])
    elif classes[class_id] == 'car':
        color = np.array([0.0, 255.0, 0.0])
    elif classes[class_id] == 'truck':
        color = np.array([0.0, 255.0, 255.0])

    thick = 1
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, thick)

    cv2.rectangle(img, (x, y - 20), (x + int((120 / 16) * len(label)), y - 5), color / 3, -1)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    overlay = img.copy()
    cv2.rectangle(overlay, (x + thick, y + thick), (x_plus_w - thick, y_plus_h - thick), color, -1)
    cv2.putText(overlay, '@The-Vheed', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    alpha = 0.1

    image = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS) / skip_factor

ret, image = fetchframe(source)

image_scale = 960 / image.shape[1]

frame_size = (int(image.shape[1] * image_scale), int(image.shape[0] * image_scale))

output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

with open(classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
scale = 0.00392

net = cv2.dnn.readNetFromDarknet(config, weights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# Uncomment the following lines to use an available NVIDIA GPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

conf_threshold = 0.5
nms_threshold = 0.4

count = -1
while count < 100000:
    try:
        print('\b' * (len(str(count)) + len(' frame(s)')), end='')
        count += 1
        print(str(count) + ' frame(s)', end='')
        # image = cv2.imread(image)

        ret, image = fetchframe(source)

        if count % skip_factor == 0:
            if not ret:
                break
            image = cv2.resize(image, (int(image.shape[1] * image_scale), int(image.shape[0] * image_scale)),
                               interpolation=cv2.INTER_AREA)

            Width = image.shape[1]
            Height = image.shape[0]

            blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            for i in indices:
                i = i
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

            output.write(image)
            cv2.imshow('Frame', image)
    except KeyboardInterrupt:
        break

# cv2.imwrite("object-detection.jpg", image)
output.release()
