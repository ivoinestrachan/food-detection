import cv2
import os
import json
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
source = './dataset/ingredients'

with open('recipes.json') as f:
    recipes = json.load(f)

def get_recipes(item):
    return [recipe["recipe"] for recipe in recipes if recipe["fruit"].lower() == item.lower()]

for filename in os.listdir(source):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        img_path = os.path.join(source, filename)
        image = cv2.imread(img_path)

        if image is not None:
            results = model(image)

            for result in results:
                for box in result.boxes:
                    class_index = int(box.cls.item())
                    detected_item = result.names[class_index]

                    recipe_suggestions = get_recipes(detected_item)
                    for recipe in recipe_suggestions:
                        print(f"food: {detected_item}, recipe suggestion: {recipe}")
