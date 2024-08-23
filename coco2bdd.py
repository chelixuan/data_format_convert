import os
import json
from tqdm import tqdm

raw_json = '/mnt/e/dataset/images/annotations/specific_categories/train_0412-0414_batch_0.json'
save_root = '/mnt/e/dataset/yolop/bdd_format/det/train/'
select_img_path = '/mnt/e/datasets/dataset/yolop/train_labeled/'
folder = 'train/train_0412-0414_batch_0/'

os.makedirs(save_root, exist_ok = True)

selected_images = os.listdir(select_img_path)
selected_images = [x for x in selected_images if '.jpg' in x]

f = open(raw_json)
info = json.load(f)

images = info['images']
annotations = info['annotations']
categories = info['categories']

cate_id2name = dict()
for t_cate in categories:
    t_cate_id = t_cate['id']
    t_cate_name = t_cate['name']
    cate_id2name[t_cate_id] = t_cate_name

for t_image in tqdm(selected_images):
    image_name = folder + t_image
    
    image = [x for x in images if x['file_name'] == image_name]
    if len(image) == 0:
        bdd_objects = []
    else:
        image_id = image[0]['id']
        img_ann = [x for x in annotations if x['image_id'] == image_id]
        bdd_objects = []
        for i in range(len(img_ann)):
            t_ann = img_ann[i]
            cate_id = t_ann['category_id']
            cate_name = cate_id2name[cate_id]

            x1, y1, w, h = t_ann['bbox']
            x2, y2 = x1 + w, y1 + h

            t_object = {
                "category": cate_name,
                "id": i,
                "attributes": {
                    "occluded": bool(0),
                    "truncated": bool(0),
                    "trafficLightColor": "none"
                },
                "box2d": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            }
            
            bdd_objects.append(t_object)

    
    name = image_name[image_name.rfind('/')+1:-4]

    if name + '.jpg' not in selected_images:
        continue

    frames = {
        'timestamp': 10000,
        'frames': [bdd_objects]
    }
    attributes =  {
        "weather": "clear",
        "scene": "swimming pool",
        "timeofday": "day"
    }

    bdd_info = dict()
    bdd_info['name'] = name
    bdd_info['frames'] = frames
    bdd_info['attributes'] = attributes

    save_path = save_root + name + '.json'
    with open(save_path, 'w') as f:
        json.dump(bdd_info, f, indent = 4)

print('\nfinished ~~ \n')

