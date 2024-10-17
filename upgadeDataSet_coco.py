import os
import json
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm
import copy

# Parâmetro para o número de aumentações por imagem
num_augmentations = 10  # Altere este valor conforme necessário

# Caminhos das pastas e arquivos
input_images_dir = 'D:\\marke_img\\data\\'
input_annotations_file = 'D:\\marke_img\\data\\result.json'
output_images_dir = 'imagens_aumentadas_coco\\'
output_annotations_file = 'anotacoes_coco_aumentadas\\annotations.json'

# Cria a pasta de saída se não existir
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_annotations_file), exist_ok=True)

# Carrega o arquivo de anotações COCO
with open(input_annotations_file, 'r') as f:
    coco_data = json.load(f)

# Inicializa os dados COCO para as imagens aumentadas
augmented_coco_data = copy.deepcopy(coco_data)
augmented_coco_data['images'] = []
augmented_coco_data['annotations'] = []

# Contadores para IDs únicos
image_id_counter = max(image['id'] for image in coco_data['images']) + 1
annotation_id_counter = max(ann['id'] for ann in coco_data['annotations']) + 1

# Organiza as anotações por image_id
annotations_by_image = {}
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(ann)

# Define as transformações
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(rotate=(-15, 15), p=0.5),
    # Adicione mais transformações se desejar
], bbox_params=A.BboxParams(
    format='coco',  # Indica que as caixas estão em pixels
    label_fields=['category_ids'],
    #min_visibility=0.3
))

# Itera sobre cada imagem e suas anotações
for image_info in tqdm(coco_data['images'], desc='Processando imagens'):
    image_id = image_info['id']
    img_filename = image_info['file_name']
    img_path = os.path.join(input_images_dir, img_filename)

    # Lê a imagem
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        print(f"Imagem {img_path} não encontrada ou não pôde ser lida.")
        continue

    height, width = image.shape[:2]

    # Atualiza as dimensões da imagem
    image_info['height'] = height
    image_info['width'] = width

    # Obtém as anotações da imagem
    anns = annotations_by_image.get(image_id, [])

    # Extrai as caixas delimitadoras e categorias
    bboxes = []
    category_ids = []
    for ann in anns:
        bbox = ann['bbox']  # [x_min, y_min, width, height] em pixels
        bboxes.append(bbox)
        category_ids.append(int(ann['category_id']))

    # Gera múltiplas aumentações
    for i in range(num_augmentations):
        # Aplica as transformações
        try:
            augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        except Exception as e:
            print(f"Erro ao aplicar transformações na imagem {img_filename}- {e}")
            continue

        # Obtém a imagem e as anotações aumentadas
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_category_ids = augmented['category_ids']

        # Verifica se ainda existem caixas válidas
        if len(augmented_bboxes) == 0:
            continue

        # Atualiza o image_info para a imagem aumentada
        new_image_info = {
            'id': image_id_counter,
            'file_name': f"{os.path.splitext(img_filename)[0]}_aug_{i}.jpg",
            'width': augmented_image.shape[1],
            'height': augmented_image.shape[0]
        }
        augmented_coco_data['images'].append(new_image_info)

        # Salva a imagem aumentada
        output_image_path = os.path.join(output_images_dir, new_image_info['file_name'])
        cv2.imwrite(output_image_path, augmented_image)
        print(f"Imagem aumentada salva em {output_image_path}")

        # Atualiza as anotações para a imagem aumentada
        for bbox, category_id in zip(augmented_bboxes, augmented_category_ids):
            x_min, y_min, width_bbox, height_bbox = bbox

            # Cria a nova anotação
            new_annotation = {
                'id': annotation_id_counter,
                'image_id': new_image_info['id'],
                'category_id': category_id,
                'bbox': [x_min, y_min, width_bbox, height_bbox],
                'area': width_bbox * height_bbox,
                'iscrowd': 0,
                'segmentation': []  # Se tiver segmentação, atualize aqui
                
            }
            augmented_coco_data['annotations'].append(new_annotation)
            annotation_id_counter += 1

        # Incrementa o contador de image_id
        image_id_counter += 1

# Salva o arquivo de anotações atualizado
with open(output_annotations_file, 'w') as f:
    json.dump(augmented_coco_data, f, ensure_ascii=False, indent=4)

print("Processo de aumentação concluído com sucesso!")

