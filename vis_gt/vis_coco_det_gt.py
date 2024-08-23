import os
import cv2
import json
import random
from tqdm import tqdm

raw_path = '/mnt/f/dataset/images_with_annotation/'
ann_path = '/mnt/f/dataset/train.json'
vis_gt_save_path = '/mnt/f/dataset/temp_vis_gt/'
os.makedirs(vis_gt_save_path, exist_ok=True)

info = json.load(open(ann_path))
images = info['images']
annotations = info['annotations']
categories = info['categories']

cate_id2cate = {}
for cate in categories:
    cate_id = cate['id']
    cate_name = cate['name']
    cate_id2cate[cate_id] = cate_name

images = random.sample(images, 100)
for image_info in tqdm(images):
    image_name = image_info['file_name']
    image_id = image_info['id']

    img = cv2.imread(raw_path + image_name)
    for t_ann in annotations:
        if t_ann['image_id'] == image_id:
            xmin, ymin, w, h = t_ann['bbox']
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmin + w), int(ymin + h)
            cate_id = t_ann['category_id']
            # if cate_id == 1:
            #     color = (0, 255, 0)
            # elif cate_id == 2:
            #     color = (0, 0, 255)

            color = [random.randint(0, 255) for i in range(3)]

            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 1)

            segmentation = t_ann['segmentation'][0]
            for index in range(0, len(segmentation), 2):
                point_x = float(segmentation[index])
                point_y = float(segmentation[index + 1])
                point_x, point_y = int(point_x), int(point_y)

                if index == 0:
                    init_x, init_y = point_x, point_y
                    start_x, start_y = point_x, point_y
                    end_x, end_y = point_x, point_y
                
                else:
                    start_x, start_y = end_x, end_y
                    end_x, end_y = point_x, point_y
                
                cv2.circle(img, (point_x, point_y), 5, color, -1)
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)

                if index == len(segmentation) - 2:
                    start_x, start_y = end_x, end_y
                    end_x, end_y = init_x, init_y

                    cv2.circle(img, (point_x, point_y), 5, color, -1)
                    cv2.line(img, (start_x, start_y), (end_x, end_y), color, 3)

    cv2.imwrite(vis_gt_save_path + image_name, img)

