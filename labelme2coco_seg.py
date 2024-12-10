import os
import cv2
import json
from tqdm import tqdm

raw_path = '/home/chelx/m2_step/dataset/images_with_annotation/'
save_path = '/home/chelx/m2_step/dataset/annotations/train_pool_without_wall.json'

# json_files = os.listdir(raw_path)
# json_files = [x for x in json_files if x[-5:] == '.json']
image_files = os.listdir(raw_path)
image_files = [x for x in image_files if x[-4:] == '.jpg']


categories = [
    {'supercategory': 'UnDrivable Area', 'id': 1, 'name': 'facade'},
    {'supercategory': 'Drivable Area', 'id': 2, 'name': 'step'},
    {'supercategory': 'Drivable Area', 'id': 3, 'name': 'floor'},
    # {'supercategory': 'Drivable Area', 'id': 4, 'name': 'wall'},
]

cate2id = {
    'facade': 1, 
    'step': 2,
    'floor': 3,
    # 'wall': 4,
}


images = []
annos = []
count_image = 0
count_anno = 0
for t_image in tqdm(image_files):
    img = cv2.imread(raw_path + t_image)
    image_height, image_width, _ = img.shape

    t_json = t_image[:-4] + '.json'
    t_image_info = {'file_name': t_image,
                    # 'height': 1088, 
                    # 'width': 1920, 
                    'height': image_height, 
                    'width': image_width, 
                    'id': count_image}
    
    images.append(t_image_info)

    if os.path.exists(raw_path + t_json):
        f = open(raw_path + t_json)
        info = json.load(f)

        manner_annos = info['shapes']
        for manner_anno in manner_annos:
            label_name = manner_anno['label']

            # 不考虑池壁 --------------------------------
            if label_name in ['wall']:
                continue
            # -----------------------------------------

            points = manner_anno['points']

            cate_id = cate2id[label_name]

            seg = []
            box_x1, box_y1 = image_width, image_height
            box_x2, box_y2 = 0, 0
            for p in points:
                x, y = p
                seg.append(x)
                seg.append(y)

                box_x1, box_y1 = min(box_x1, x), min(box_y1, y)
                box_x2, box_y2 = max(box_x2, x), max(box_y2, y)
            
            anno = {'segmentation': [seg], 
                    'area': 0, 
                    'iscrowd': 0, 
                    'image_id': count_image, 
                    # 'bbox': [0, 0, 0, 0], 
                    'bbox': [box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1], 
                    'category_id': cate_id, 
                    'id': count_anno}

            count_anno += 1
            annos.append(anno)
    # else:
    #     anno = {'segmentation': [[]], 
    #             'area': 0, 
    #             'iscrowd': 0, 
    #             'image_id': count_image, 
    #             # 'bbox': [473.07, 395.93, 38.65, 28.67],
    #             'bbox': [0, 0, 0, 0], 
    #             'category_id': 0, 
    #             'id': count_anno}
    #     count_anno += 1
    #     annos.append(anno)
    
    count_image += 1

coco = dict()
coco['images'] = images
coco['categories'] = categories
coco['annotations'] = annos


with open(save_path, 'w') as f:
    json.dump(coco, f, indent = 4)

print('finished ~~ \n')
        

