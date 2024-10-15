import albumentations as A
import cv2
import numpy as np
import json 
import copy

num_augmentations = 10

input_images_dir = 'data'
input_annotations_file = 'data\\result.json'
output_images_dir = 'dataset\images'
output_annotations_file ='dataset\\result.json'

with open(input_annotations_file) as f:
    data = json.load(f)

augmenteds_coco = copy.deepcopy(data)
augmenteds_coco['images'] = []
augmenteds_coco['annotations'] = []

image_id_counter = ( max([image['id'] for image in data['images']]) + 1 )
annotation_id_counter = ( max([annotation['id'] for annotation in data['annotations']]) + 1 )


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='coco', 
                            label_fields=['category_id'],
                            min_visibility=0.8, 
                            check_each_transform=False,
                            clip=True
                            ))

for image in data['images']:
    image_id = image['id']
    image_file = image['file_name']
    image_path = f'{input_images_dir}/{image_file}'
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    height, width, _ = image_data.shape
    annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]
    bboxes = [annotation['bbox'] for annotation in annotations]
    categories = [annotation['category_id'] for annotation in annotations]
    for i in range(num_augmentations):
        try:
            augumented = transform(image=image_data, bboxes=bboxes, category_id=categories)
        except:
            continue
        augmented_image = augumented['image']
        augmented_bboxes = augumented['bboxes']
        augmented_categories = augumented['category_id']
        new_image_info = {
            'id': image_id_counter,
            'file_name': f'{image_id_counter}.jpg',
            'height': height,
            'width': width
        }
        augmenteds_coco['images'].append(new_image_info)
        for bbox, category in zip(augmented_bboxes, augmented_categories):

            new_annotation_info = {
                'id': annotation_id_counter,
                'image_id': image_id_counter,
                'category_id': category,
                'bbox': bbox,
                'segmentation': [],
                'ignore': 0,
                'iscrowd': 0,
            }
            augmenteds_coco['annotations'].append(new_annotation_info)
            annotation_id_counter += 1
        cv2.imwrite(f'{output_images_dir}/{image_id_counter}.jpg', cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        image_id_counter += 1

with open(output_annotations_file, 'w') as f:
    json.dump(augmenteds_coco, f, indent=4)



def visualizar(image_id):
    map_category_id_to_name = {0:'Questao', 2:'Topico', 1:'Texto'}
    image_file = [image['file_name'] for image in augmenteds_coco['images'] if image['id'] == image_id][0]
    image_path = f'{output_images_dir}/{image_file}'
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    annotations = [annotation for annotation in augmenteds_coco['annotations'] if annotation['image_id'] == image_id]
    bboxes = [annotation['bbox'] for annotation in annotations]
    categories = [annotation['category_id'] for annotation in annotations]
    for bbox, category in zip(bboxes, categories):
        x, y, w, h = bbox
        print(category, x, y, w, h)
        cv2.rectangle(image_data, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        cv2.putText(image_data,map_category_id_to_name[category], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    #visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()


#visualizar(41)
            
