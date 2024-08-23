import os
import cv2
from tqdm import tqdm

image_root = '/home/chelx/dataset/QR_1080p/retina_format/val/images/'
txt_path = '/home/chelx/dataset/QR_1080p/retina_format/val/label.txt'
save_vis_path = '/home/chelx/dataset/QR_1080p/retina_format/vis_val_gt/'
os.makedirs(save_vis_path, exist_ok=True)

f = open(txt_path)
info = f.read().split('\n')
info = info[:-1]
image_index = [i for i in range(0, len(info), 2)]

import random
image_index = random.sample(image_index, min(100, len(image_index)))

for index in tqdm(image_index):
    image = info[index][2:]
    img = cv2.imread(image_root + image)

    tag_info = info[index + 1].split()
    box_x1, box_y1, box_w, box_h = tag_info[0:4]
    p1_x, p1_y = tag_info[4], tag_info[5]
    p2_x, p2_y = tag_info[7], tag_info[8]
    p3_x, p3_y = tag_info[10], tag_info[11]
    p4_x, p4_y = tag_info[13], tag_info[14]
    
    box_x1, box_y1, box_x2, box_y2 = int(float(box_x1)), int(float(box_y1)), int(float(box_x1)+float(box_w)), int(float(box_y1)+float(box_h))
    p1_x, p1_y = int(float(p1_x)), int(float(p1_y))
    p2_x, p2_y = int(float(p2_x)), int(float(p2_y))
    p3_x, p3_y = int(float(p3_x)), int(float(p3_y))
    p4_x, p4_y = int(float(p4_x)), int(float(p4_y))


    cv2.circle(img, (p1_x, p1_y), 1, (0, 0, 255), -1)
    cv2.circle(img, (p2_x, p2_y), 1, (0, 255, 255), -1)
    cv2.circle(img, (p3_x, p3_y), 1, (255, 0, 255), -1)
    cv2.circle(img, (p4_x, p4_y), 1, (0, 255, 0), -1)

    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (100, 100, 100), 1)  

    image_name = image[image.rfind('/')+1:]
    cv2.imwrite(save_vis_path + image_name, img)

