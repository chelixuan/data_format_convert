import os
import json
import cv2
from tqdm import tqdm

import numpy as np

json_path = "/media/cv/dataset/seg_images/coco_annotations/standard_coco_annos/val_batch01_lyon2024_202409.json"
image_root = "/media/cv/dataset/seg_images/images/val/"
seg_img_save_path = "/media/cv/dataset/seg_images/semantic_data/labelTrainIdsPng_SingleChannel/val/"
os.makedirs(seg_img_save_path, exist_ok=True)

CATES = ["person", "cat", "dog", "bicycle", "car", "bus",]
COLORS = [[50, 192, 164], [128, 92, 112], [125, 164, 100], [128, 232, 80], [164, 90, 164], [200, 50, 150]]

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

    # mask_img = np.zeros([image_height, image_width, 3], dtype=np.uint8)
    # mask_img = np.full((image_height, image_width, 3), 255, dtype=np.uint8)

    mask_img = np.full((image_height, image_width), 255, dtype=np.uint8)

    for t_ann in annos:
        if t_ann["image_id"] == image_id:
            cate_id = t_ann["category_id"]

            seg = t_ann["segmentation"][0]
            seg_point = []
            for i in range(0, len(seg), 2):
                point_x = seg[i]
                point_y = seg[i + 1]
                seg_point.append([point_x, point_y])
            start_point = seg_point[0]
            seg_point.append(start_point)

            seg_point = np.array([seg_point], dtype=np.int32)

            # color = COLORS[cate_id - 1]
            # color = [cate_id, cate_id, cate_id]
            color = cate_id
            cv2.fillPoly(mask_img, seg_point, color)
            

    cv2.imwrite(seg_img_save_path + save_name, mask_img)


