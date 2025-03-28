import os
import cv2
import json
import numpy as np
from scipy.io import savemat
from tqdm import tqdm
import shutil
 
json_root = "/media/wybj/cv/dataset/"

# json_path = json_root + "seg_images/coco_annotations/standard_coco_annos/val_batch01_lyon2024_202409.json"
# image_root = "/media/wybj/cv/dataset/seg_images/images/val/"

json_path = json_root + "seg_images/coco_annotations/standard_coco_annos/train_batch00_tianjin_202408.json"
# json_path = json_root + "seg_images/coco_annotations/standard_coco_annos/train_batch01_lyon2024_202409-202411.json"
image_root = "/media/wybj/cv/dataset/seg_images/images/train/"


mat_save_path = "/media/wybj/cv/dataset/seg_images/semantic_data/COCOStuff_mat_format/all_annotations/"
os.makedirs(mat_save_path, exist_ok=True)
image_save_path = "/media/wybj/cv/dataset/seg_images/semantic_data/COCOStuff_mat_format/all_images/"
os.makedirs(image_save_path, exist_ok=True)
 
info = json.load(open(json_path))
categories = info["categories"]
annos = info["annotations"]
images = info["images"]

category_map = {cat['id']: cat['name'] for cat in categories}
# image_id2name = {image["id"]: image["file_name"][image["file_name"].rfind("/")+1:] for image in images}

for image in tqdm(images):
    file_name = image["file_name"]
    image_height, image_width = image["height"], image["width"]
    image_id = image["id"]

    image_name = file_name[file_name.rfind("/")+1:]
    save_mat_name = image_name[:-4] + ".mat"
    shutil.copy(image_root + file_name, image_save_path + image_name)
    
    S = np.zeros([image_height, image_width], dtype=np.uint8)

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

            cv2.fillPoly(S, seg_point, cate_id)
    mat_content = dict()
    mat_content["S"] = S
    savemat(mat_save_path + save_mat_name, mat_content)

