import cv2
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

source = './dataset/ingredients'

for filename in os.listdir(source):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
        img_path = os.path.join(source, filename)

        image = cv2.imread(img_path)
        if image is not None:
            results = model(image)
