from xml.dom.minidom import parse
import json
import os
import cv2

import shutil
from tqdm import tqdm
from collections import Counter

def readXML(path):
    domTree = parse(path)
    rootNode = domTree.documentElement
    #print(rootNode.nodeName)
    
    size = rootNode.getElementsByTagName("size")[0]
    width = size.getElementsByTagName("width")[0]
    width = width.childNodes[0].data
    height = size.getElementsByTagName("height")[0]
    height = height.childNodes[0].data


    objects = rootNode.getElementsByTagName("object")
    
    bboxes = []
    labels = []

    for object in objects:
        name = object.getElementsByTagName("name")[0]
        name = name.childNodes[0].data
        #print(name.nodeName, ":", name.childNodes[0].data)
        
        bbox = object.getElementsByTagName("bndbox")[0]
        xmin = bbox.getElementsByTagName("xmin")[0]
        x_min = xmin.childNodes[0].data
        ymin = bbox.getElementsByTagName("ymin")[0]
        y_min = ymin.childNodes[0].data
        xmax = bbox.getElementsByTagName("xmax")[0]
        x_max = xmax.childNodes[0].data
        ymax = bbox.getElementsByTagName("ymax")[0]
        y_max = ymax.childNodes[0].data
        
        #print(name)
        #print(x_min, y_min, x_max, y_max)
        labels.append(name)
        bbox = [x_min, y_min, x_max, y_max]
        bboxes.append(bbox)
    return width,height,labels,bboxes

cate = [
        {
            "supercategory": "charging_station",
            "id": 1,
            "name": "charging_station"
        },
]

count_img = 0
count_ann = 0

images = []
ann = []

raw_path = '/mnt/e/datasets/dataset/images/val/trash_0412-0414_batch_0/'
xmls = os.listdir(raw_path)
xmls = [x for x in xmls if '.xml' in x]
print(len(xmls))

# save_root = '/mnt/e/datasets/dataset/images/train/k210_charging/'
# os.makedirs(save_root, exist_ok=True)

# all_labels = ['other_trash', 'stain', 'stone', 'coplanar', 'protrusion', 'drain', 'leaf', 'other_robot', \
#               'protrudent_drain', 'charging_station', 'other_equipment', 'sandpile', 'scree', 'lamp']

all_labels = []
for xml in tqdm(xmls):
    width, height, labels, bboxes = readXML(raw_path + xml)

    all_labels += labels

    if 'charging_station' in labels:
        image_name = xml[:-4] + '.jpg'
        # shutil.copy(raw_path + image_name, save_root + image_name)

        image_id = count_img
        count_img += 1

        image = {
            "file_name": image_name,
            "height": int(float(height)),
            "width": int(float(width)),
            "id": image_id
        }
        images.append(image)

        count_charging = labels.count('charging_station')
        if count_charging > 1:
            print(count_charging, image_name)
            print()

'''

        for i in range(len(labels)):
            t_label = labels[i]
            if t_label == 'charging_station':
                xmin, ymin, xmax, ymax = bboxes[i]
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                w, h = xmax - xmin, ymax - ymin

                t_ann = {
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [
                        xmin,
                        ymin,
                        w,
                        h
                    ],
                    "category_id": 1,
                    "id": count_ann
                }
                ann.append(t_ann)
                count_ann += 1



coco = dict()
coco['images'] = images
coco['annotations'] = ann
coco['categories'] = cate

charging_json = '/mnt/e/datasets/dataset/images/train/k210_charging_train.json'
with open(charging_json, 'w') as f:
    json.dump(coco, f, indent=4)
    print('success ~~ \n')

'''

