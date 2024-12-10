import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm

coco_path = '/home/chelx/dataset/seg_images/coco_annotations/train_20241118.json'
image_root = '/home/chelx/dataset/seg_images/images/train/'
save_path = '/home/chelx/dataset/seg_images/images/temp_vis_train/'
os.makedirs(save_path, exist_ok=True)

info = json.load(open(coco_path))
images = info['images']
cate = info['categories']
anno = info['annotations']

txt_root_path = "/home/chelx/dataset/seg_images/yolo_txt_annos/val/val_batch_01_202409_lyon/"
image_root = "/home/chelx/dataset/seg_images/images/val/val_batch_01_202409_lyon/"
save_path = "/home/chelx/dataset/seg_images/yolo_txt_annos/vis_val_gt/"
os.makedirs(save_path, exist_ok=True)

images = os.listdir(image_root)
images = random.sample(images, min(len(images), 50))

cates = ["step", "facade", "floor", "wall"]

def plot_seg_contour(img, seg_point, color, image_width, image_height):
    x, y = [], []
    for index in range(0, len(seg_point), 2):
        point_x = float(seg_point[index])
        point_y = float(seg_point[index + 1])

        point_x, point_y = int(point_x), int(point_y)

        point_x, point_y = min(point_x, image_width), min(point_y, image_height)
        point_x, point_y = max(0, point_x), max(0, point_y)

        x.append(point_x)
        y.append(point_y)
        
        if index == 0:
            init_x, init_y = point_x, point_y
            start_x, start_y = point_x, point_y
            end_x, end_y = point_x, point_y

        # elif index == int(len(seg_point)/2 - 1):
        #     start_x, start_y = end_x, end_y
        #     end_x, end_y = init_x, init_y
        
        else:
            start_x, start_y = end_x, end_y
            end_x, end_y = point_x, point_y
        
        
        cv2.circle(img, (point_x, point_y), 5, color, -1)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)

        if index == len(seg_point) - 2:
            start_x, start_y = end_x, end_y
            end_x, end_y = init_x, init_y

            cv2.circle(img, (point_x, point_y), 5, color, -1)
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)
    
    # return img
    return img, int(np.mean(x)), int(np.mean(y))
        

for t_image in tqdm(images):
    img = cv2.imread(image_root + t_image)
    image_height, image_width, _ = img.shape

    txt_path = txt_root_path + t_image[:-4] + ".txt"
    with open(txt_path, "r") as f:
        info = f.read().split("\n")[:-1]
        for t_info in info:
            t_info = t_info.split(" ")

            t_cate_id = int(float(t_info[0]))
            t_cate_name = cates[t_cate_id]

            seg = []
            for i in range(1, len(t_info)-1, 2):
                point_x, point_y = float(t_info[i]), float(t_info[i+1])
                x, y = point_x * image_width, point_y * image_height

                color = [random.randint(0, 255) for _ in range(3)]

                seg.append(x)
                seg.append(y)

            img, mean_x, mean_y = plot_seg_contour(img, seg, color, image_width, image_height)
            cv2.putText(img, t_cate_name, (mean_x, mean_y), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)


    cv2.imwrite(save_path + t_image, img)

