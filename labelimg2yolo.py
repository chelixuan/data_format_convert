import os
import cv2
from tqdm import tqdm
from xml.dom import minidom

raw_path = '/mnt/e/datasets/images/train/train_batch_0_0412-0414/'
image_path = '/mnt/e/datasets/multi_task_dataset/images/train/'
det_txt_save_path = '/mnt/e/dataset/multi_task_dataset/all_categories-0131/all-detection-object/labels/train/'

os.makedirs(det_txt_save_path, exist_ok=True)

raw_images = [x for x in os.listdir(raw_path) if x[-4:]=='.jpg']
images = [x for x in os.listdir(image_path) if x[-4:] == '.jpg']

categories = {
     'person' : 0,
     'bus': 1,
     'car': 1,
     'bicycle' : 2,
}

for t_image in tqdm(images):
    if t_image not in raw_images:
        print('missing : ', t_image)
        print()
        continue

    img = cv2.imread(image_path + t_image)
    image_height, image_width, _ = img.shape
    xml_name = t_image[:-4] + '.xml'

    if not os.path.exists(raw_path + xml_name):
        print('annotation missing : ', xml_name)
        continue

    txt_info = []

    info = minidom.parse(raw_path + xml_name)
    rootNode = info.documentElement
    for i in range(len(rootNode.childNodes)):
        node = rootNode.childNodes[i]
        if node.nodeName == 'object':
            category_name = node.getElementsByTagName('name')
            category_name = category_name[0].childNodes[0].data
            # print('category : ', category_name)

            bbox = node.getElementsByTagName('bndbox')[0]
            xmin = bbox.getElementsByTagName('xmin')[0]
            xmin = xmin.childNodes[0].data
            ymin = bbox.getElementsByTagName('ymin')[0]
            ymin = ymin.childNodes[0].data
            xmax = bbox.getElementsByTagName('xmax')[0]
            xmax = xmax.childNodes[0].data
            ymax = bbox.getElementsByTagName('ymax')[0]
            ymax = ymax.childNodes[0].data

            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            # print('xmin, ymin, xmax, ymax = ', xmin, ymin, xmax, ymax)
            if category_name in list(categories.keys()):
                cate_id = categories[category_name]
                x_center, y_center, w, h = 0.5*(xmin + xmax), 0.5*(ymin + ymax), xmax - xmin, ymax - ymin
                point_xc, point_yc, point_w, point_h = x_center/image_width, y_center/image_height, w/image_width, h/image_height

                box_info = str(cate_id) + ' ' + str(point_xc) + ' ' + str(point_yc) + ' ' + str(point_w) + ' '+ str(point_h) + '\n'
                txt_info.append(box_info)
    
    if len(txt_info) == 0:
        continue

    txt_name = t_image[:-4] + '.txt'
    with open(det_txt_save_path + txt_name, 'w') as f:
        for t_info in txt_info:
            f.write(t_info)

print('\nsuccess ~~ \n')

