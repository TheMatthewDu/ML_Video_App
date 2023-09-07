import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import AlexNet

import cv2 as cv
import json

import time

import numpy as np
import pandas as pd
import torch

col = [[0, 255, 0], [0, 0, 255]]

model1 = torch.hub.load('../yolov5', 'custom', path='weights/best.pt', source='local')
model2 = AlexNet(num_classes=2)
model2.load_state_dict(torch.load('../weights/alexnet_weights.pth', map_location=torch.device('cpu')))
cap = cv.VideoCapture(0)
while True:
    _, img = cap.read()
    show = np.zeros((480, 640, 3), dtype=np.uint8)

    dx, dy = 480 - img.shape[0], 640 - img.shape[1]
    show[dx // 2:img.shape[0] + dx // 2, dy // 2:img.shape[1] + dy // 2, :] = img

    result = model1(show)
    pts = result.pandas().xyxy[0]

    for item in pts.iterrows():
        mask = show[int(item[1]['ymin']):int(item[1]['ymax']), int(item[1]['xmin']):int(item[1]['xmax']), :]
        mask = cv.resize(mask, None, fx=100 / mask.shape[1], fy=100 / mask.shape[0])

        result = model2(torch.Tensor([mask.T]))
        print(result, torch.argmax(result))

    cv.imshow("img", show)
    key = cv.waitKey(1)
    if key == ord('q'):
        break


