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

images = random.sample(images, min(len(images), 300))
# images = [
#     {
#             "file_name": "train_batch_00_tianjin_202408/lm_654.jpg",
#             "height": 1080,
#             "width": 1920,
#             "id": 4534
#         },
# ]

# COLORS = np.random.randint(0, 255, size=(len(cate), 3))
# COLORS = [
#     [116, 53, 189],
#     [17, 227, 120],
#     [117, 108, 22],
#     [249, 242, 214],
#     [44, 190, 18],
#     [133, 170, 10],
#     [108, 231, 39],
#     [1, 205, 218],
# ]

id2cate = dict()
for t_cate in cate:
    t_cate_name = t_cate['name']
    t_cate_id = t_cate['id']
    id2cate[t_cate_id] = t_cate_name

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
    file_name = t_image['file_name']
    image_name = file_name[file_name.rfind('/')+1:]
    image_id = t_image['id']
    image_width, image_height = t_image['width'], t_image['height']

    # print('image_path : ', image_root + file_name)
    img = cv2.imread(image_root + file_name)

    for t_ann in anno:
        if t_ann['image_id'] == image_id:
            cate_id = t_ann['category_id']
            cate_name = id2cate[cate_id]
            # color = np.random.randint(0, 255, size=(len(cate), 3))
            color = [random.randint(0, 255) for _ in range(3)]

            # if 'bbox' in t_ann:
            #     x1, y1, w, h = t_ann['bbox']
            #     xmin, ymin = int(x1), int(y1)
            #     xmax, ymax = int(x1 + w), int(y1 + h)

            #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)  
            #     cv2.putText(img, cate_name, (xmin, ymin - 2), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)


            seg = t_ann['segmentation'][0]
            img, mean_x, mean_y = plot_seg_contour(img, seg, color, image_width, image_height)

            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)  
            # cv2.putText(img, cate_name, (xmin, ymin - 2), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img, cate_name, (mean_x, mean_y), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    # print('img.shape : ', img.shape)
    # print('save_path : ', save_path + image_name)

    cv2.imwrite(save_path + image_name, img)

