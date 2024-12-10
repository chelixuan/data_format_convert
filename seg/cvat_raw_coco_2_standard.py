import os
import json
from tqdm import tqdm

raw_json = '/home/chelx/dataset/Lyon2024/seg_images/raw_annotation/instances_Train.json'
save_json_path = '/home/chelx/dataset/Lyon2024/seg_images/annotations/'
os.makedirs(save_json_path, exist_ok=True)

image_save_root = '/home/chelx/dataset/seg_images/'

train_folder = 'train_batch_01_202409_lyon/'
val_folder = 'val_batch_01_202409_lyon/'

train_image_root = image_save_root + 'train/' + train_folder
os.makedirs(train_image_root, exist_ok=True)
val_image_root = image_save_root + 'val/' + val_folder
os.makedirs(val_image_root, exist_ok=True)
annotation_save_root = image_save_root + 'annotations/'
os.makedirs(annotation_save_root, exist_ok=True)

raw_info = json.load(open(raw_json))
raw_images = raw_info['images']
raw_cates = raw_info['categories']
raw_annos = raw_info['annotations']

'''
raw_cates =  [{'id': 1, 'name': '台阶平面', 'supercategory': ''}, 
              {'id': 2, 'name': '台阶立面', 'supercategory': ''}, 
              {'id': 3, 'name': '平池底', 'supercategory': ''}, 
              {'id': 4, 'name': '斜坡池底', 'supercategory': ''}, 
              {'id': 5, 'name': '立面池壁', 'supercategory': ''}, 
              {'id': 6, 'name': '斜面池壁', 'supercategory': ''}]
'''

cates = [
    {'id': 1, 'name': 'floor', 'supercategory': ''}, # 3
    {'id': 2, 'name': 'wall', 'supercategory': ''}, # 5
]

train_images, train_annos = [], []
train_image_id, train_anno_id = 0, 0
val_images, val_annos = [], []
val_image_id, val_anno_id = 0, 0


print('split_&_write images info : ')
train_image_id_convert, val_image_id_convert = {}, {}
raw_train_ids, raw_val_ids = [], []
for t_image in tqdm(raw_images):
    raw_image_id = t_image['id']
    file_name = t_image['file_name']
    image_name = file_name[file_name.rfind('/')+1:]
    if 'train' in file_name:
        image = {'id': train_image_id, 
                 'width': t_image['width'], 
                 'height': t_image['height'], 
                 'file_name': train_folder + image_name, 
                 }
        train_images.append(image)

        train_image_id_convert[raw_image_id] = train_image_id
        train_image_id += 1
        raw_train_ids.append(raw_image_id)
    elif 'val' in file_name:
        image = {'id': val_image_id, 
                 'width': t_image['width'], 
                 'height': t_image['height'], 
                 'file_name': val_folder + image_name, 
                 }
        val_images.append(image)

        val_image_id_convert[raw_image_id] = val_image_id
        val_image_id += 1
        raw_val_ids.append(raw_image_id)
    else:
        print('wrong !!! ')
        print('file_name = ', file_name)
        exit()

'''
anno =  {'id': 1, 'image_id': 1, 'category_id': 5, 
         'segmentation': [[838.42, 462.52, 1413.06, 718.81,], 
         'iscrowd': 0, 'attributes': {'occluded': False}}

'''
print('split_&_write annotations info : ')
for t_ann in tqdm(raw_annos):
    t_cate_id = t_ann['category_id']
    if t_cate_id in [3, 5]:
        if t_cate_id == 3:
            t_cate_id_new = 1
        elif t_cate_id == 5:
            t_cate_id_new = 2

        raw_image_id = t_ann['image_id']
        if raw_image_id in raw_train_ids:
            image_id = train_image_id_convert[t_ann['image_id']]
            t_ann_new = {
                'id': train_anno_id, 
                'image_id': image_id, 
                'category_id': t_cate_id_new, 
                'segmentation': t_ann['segmentation'],
            }

            train_annos.append(t_ann_new)
            train_anno_id += 1
        elif raw_image_id in raw_val_ids:
            image_id = val_image_id_convert[t_ann['image_id']]
            t_ann_new = {
                'id': val_anno_id, 
                'image_id': image_id, 
                'category_id': t_cate_id_new, 
                'segmentation': t_ann['segmentation'],
            }

            val_annos.append(t_ann_new)
            val_anno_id += 1

train_info = dict()
train_info['images'] = train_images
train_info['categories'] = cates
train_info['annotations'] = train_annos

train_json_path = annotation_save_root + 'train_batch_01_202409_lyon.json'
with open(train_json_path, 'w') as f:
    json.dump(train_info, f, indent=4)
print(f'train_json has been writen to : {train_json_path} \n')

val_info = dict()
val_info['images'] = val_images
val_info['categories'] = cates
val_info['annotations'] = val_annos

val_json_path = annotation_save_root + 'val_batch_01_202409_lyon.json'
with open(val_json_path, 'w') as f:
    json.dump(val_info, f, indent=4)
print(f'val_json has been writen to : {val_json_path} \n')

