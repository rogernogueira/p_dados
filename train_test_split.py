from ultralytics.data.utils import autosplit
import os
import shutil

dir_dataset = 'D:\yolo_layout\yolo_data_augmented'
autosplit(dir_dataset, weights=(0.8, 0.2, 0.0), annotated_only=True)
DIRS = ['train', 'val']
for d in DIRS:
    os.makedirs(os.path.join('D:\yolo_layout\yolo_data_augmented', d, 'images'), exist_ok=True)
    os.makedirs(os.path.join('D:\yolo_layout\yolo_data_augmented', d, 'labels'), exist_ok=True)

for d in DIRS:
    with open(f'autosplit_{d}.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        img_path = line 
        label_path = img_path.replace('images', 'labels').replace('.png', '.txt')
        shutil.copy(img_path, os.path.join(dir_dataset, d, 'images'))
        shutil.copy(label_path, os.path.join(dir_dataset, d, 'labels'))
