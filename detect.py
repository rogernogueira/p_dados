from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

model = YOLO('D:\yolo_layout\model\yolov10x_best.pt')  # Load model
results = model('D:\yolo_layout\\00ff6644-9.png')  # Inference

results[0].show()

 # Show results

