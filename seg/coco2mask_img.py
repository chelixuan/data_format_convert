import os
import json
import cv2
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

json_path = "/home/chelx/dataset/seg_images/coco_annotations/standard_coco_annos/train_batch00_tianjin_202408.json"
image_root = "/home/chelx/dataset/seg_images/images/train/"

# category_id: 3 -- floor
floor_save_path = "/home/chelx/dataset/seg_images/semantic_seg_anno/floor/train_batch_00_tianjin_202408/"
os.makedirs(floor_save_path, exist_ok=True)
# category_id: 5 -- wall
wall_save_path = "/home/chelx/dataset/seg_images/semantic_seg_anno/wall/train_batch_00_tianjin_202408/"
os.makedirs(wall_save_path, exist_ok=True)

f = open(json_path)
info = json.load(f)
images = info["images"]
cate = info["categories"]
annos = info["annotations"]

for t_image in tqdm(images):
    file_name = t_image["file_name"]
    image_name = file_name[file_name.rfind("/") + 1 : ]
    save_name = image_name[:-4] + ".png"
    image_id = t_image["id"]

    img = cv2.imread(image_root + file_name)
    image_height, image_width, _ = img.shape

    floor = np.zeros([image_height, image_width], dtype=np.uint8)
    wall = np.zeros([image_height, image_width], dtype=np.uint8)

    for t_ann in annos:
        if t_ann["image_id"] == image_id and t_ann["category_id"] in [3, 5]:
            seg = t_ann["segmentation"][0]
            seg_point = []
            for i in range(0, len(seg), 2):
                point_x = seg[i]
                point_y = seg[i + 1]
                seg_point.append([point_x, point_y])
            start_point = seg_point[0]
            seg_point.append(start_point)

            seg_point = np.array([seg_point], dtype=np.int32)

            if t_ann["category_id"] == 3:
                # cv2.polylines(floor, seg_point, True, 255, 1)
                cv2.fillPoly(floor, seg_point, 255)
            elif t_ann["category_id"] == 5:
                # cv2.polylines(wall, seg_point, True, 255, 1)
                cv2.fillPoly(wall, seg_point, 255)

    cv2.imwrite(floor_save_path + save_name, floor)
    cv2.imwrite(wall_save_path + save_name, wall)


