import os
import cv2
import random
from tqdm import tqdm

image_path = '/mnt/e/datasets/multi_task_dataset/images/val/'
det_txt_path = '/mnt/e/datasets/multi_task_dataset/detection-object/labels/val/'
vis_save_path = '/mnt/e/datasets/multi_task_dataset/vis_part_val_det/'
os.makedirs(vis_save_path, exist_ok=True)

images = os.listdir(image_path)
images = random.sample(images, 50)

for image_name in tqdm(images):
    txt_name = image_name[:-4] + '.txt'
    img = cv2.imread(image_path + image_name)
    image_height, image_width, _ = img.shape

    if (not os.path.exists(det_txt_path + txt_name)) or (not os.path.exists(det_txt_path + txt_name)):
        continue

    with open(det_txt_path + txt_name, 'r') as f:
        det_info = f.read()
        det_info = det_info.split('\n')
        det_info = [x for x in det_info if len(x) > 0]

    for t_info in det_info:
        t_info = t_info.split(' ')

        # color = [random.randint(0, 255) for _ in range(3)]
        color = (0, 0, 255)
        cate_id, xc, yc, w, h = int(float(t_info[0])), float(t_info[1]), float(t_info[2]), float(t_info[3]), float(t_info[4])
        xc, yc, w, h = image_width*xc, image_height*yc, image_width*w, image_height*h
        x1, y1, x2, y2 = xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.imwrite(vis_save_path+image_name, img)

print('\nsuccess ~~ \n')
