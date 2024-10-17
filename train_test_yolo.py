from ultralytics.data.utils import autosplit
import os
import shutil
dataset_path = 'D:\\marke_img\\yolo_data_augmented\\'
dirs =  autosplit(dataset_path, weights=(0.8, 0.1, 0.1),annotated_only=True )
DIRS = ['train', 'val', 'test']
for d in DIRS:
    os.makedirs(os.path.join(dataset_path, d, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, d, 'labels'), exist_ok=True)

for d in DIRS:
    with open(f'autosplit_{d}.txt', 'r') as f:
    # ler todas linhas do arquivo
        lines = f.readlines()
        for line in lines:
            # remover quebra de linha
            line = line.strip()
            # criar o caminho completo para a imagem
            img_path = line
            label_path = line.replace('images', 'labels').replace('.png', '.txt')
            shutil.copy(img_path, os.path.join(dataset_path, d, 'images'))
            shutil.copy(label_path, os.path.join(dataset_path, d, 'labels'))

           