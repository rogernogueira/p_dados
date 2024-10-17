import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

# Caminho para as pastas de imagens e labels
img_dir = 'yolo_data\images'
label_dir = 'yolo_data\labels'
img_dir_out = 'yolo_data_augmented\images'
label_dir_out = 'yolo_data_augmented\labels'

num_augmented_images = 10

# Cria a pasta de saída se não existir
os.makedirs(img_dir_out, exist_ok=True)
os.makedirs(label_dir_out, exist_ok=True)

# Transformações de augmentação
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Aplicar as transformações
for i in range(num_augmented_images):
    for img_name in glob.glob(f"{img_dir}/*.png"):
        img_path =  img_name
        label_path = img_name.replace('.png', '.txt')
        label_path = label_path.replace('images', 'labels')
        img_name = os.path.basename(img_path)

        # Ler a imagem
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # Ler os labels (formato YOLO)
        with open(label_path, 'r') as f:
            bboxes = []
            class_labels = []
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_id))

        # Aplicar as transformações
        try:
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        except:
            continue
        transformed_img = augmented['image']
        transformed_bboxes = augmented['bboxes']
        # Salvar a imagem transformada
        img_name = f"img_{i}_{img_name}"
        transformed_img_path = os.path.join(img_dir_out, img_name)
        cv2.imwrite(transformed_img_path, transformed_img)

        # Salvar os novos labels
        transformed_label_path = os.path.join(label_dir_out, img_name.replace('.png', '.txt'))
        with open(transformed_label_path, 'w') as f:
            for class_id, yolo_bbox in zip(class_labels, transformed_bboxes):
                f.write(f"{class_id} " + " ".join(map(str, yolo_bbox)) + "\n")
