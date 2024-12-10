import os
import cv2
import shutil
from tqdm import tqdm

raw_root = '/home/wybj/chelx/dataset/M3_dataset/TAG/'
save_root = '/home/wybj/chelx/dataset/M3_dataset/yolo_format/'
os.makedirs(save_root, exist_ok=True)

# val
folders = ['01', '02', '08']
sub_set = 'val/'

# train
# folders = ['03', '04', '05', '06', '07']
# sub_set = 'train/'

for i in range(len(folders)):
    folder = folders[i]
    print(f'{i+1} / {len(folders)} : ', folder)
    path = raw_root + folder + '/'
    image_path = path + 'image/'
    txt_path = path + 'point/'

    image_save_path = save_root + 'images/' + sub_set
    txt_save_path = save_root + 'labels/' + sub_set
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(txt_save_path, exist_ok=True)

    images = os.listdir(image_path)
    for image in tqdm(images):
        img = cv2.imread(image_path + image)
        image_height, image_width, _ = img.shape

        txt_name = image[:-4] + '.txt'
        f = open(txt_path + txt_name, 'r')
        info = f.read().split('\n')
        point_info = info[1].strip().split(' ')

        point1_x, point1_y = float(point_info[0]), float(point_info[1])
        point2_x, point2_y = float(point_info[2]), float(point_info[3])
        point3_x, point3_y = float(point_info[4]), float(point_info[5])
        point4_x, point4_y = float(point_info[6]), float(point_info[7])

        box_x1 = min(point1_x, point2_x, point3_x, point4_x)
        box_y1 = min(point1_y, point2_y, point3_y, point4_y)
        box_x2 = max(point1_x, point2_x, point3_x, point4_x)
        box_y2 = max(point1_y, point2_y, point3_y, point4_y)

        box_cx, box_cy = (box_x1 + box_x2) / 2, (box_y1 + box_y2) / 2
        box_w, box_h = box_x2 - box_x1, box_y2 - box_y1

        yolo_cx, yolo_cy, yolo_w, yolo_h = box_cx / image_width, box_cy / image_height, box_w / image_width, box_h / image_height
        yolo_p1_x, yolo_p1_y = point1_x / image_width, point1_y / image_height
        yolo_p2_x, yolo_p2_y = point2_x / image_width, point2_y / image_height
        yolo_p3_x, yolo_p3_y = point3_x / image_width, point3_y / image_height
        yolo_p4_x, yolo_p4_y = point4_x / image_width, point4_y / image_height

        yolo_info = f'0 {yolo_cx} {yolo_cy} {yolo_w} {yolo_h} {yolo_p1_x} {yolo_p1_y} 2.00000 {yolo_p2_x} {yolo_p2_y} 2.00000 {yolo_p3_x} {yolo_p3_y} 2.00000 {yolo_p4_x} {yolo_p4_y} 2.00000'

        shutil.copy(image_path + image, image_save_path + image)
        with open(txt_save_path + txt_name, 'w') as f2:
            f2.write(yolo_info)
        
