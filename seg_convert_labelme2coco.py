import os
import cv2
import json
from tqdm import tqdm

raw_path = '/mnt/f/dataset/images_with_annotation/'
json_files = [x for x in os.listdir(raw_path) if x[-5:] == '.json']

save_path = '/mnt/f/dataset/train.json'


images = []
annotations = []
categories = [{"supercategory": "drivable", "id": 1, "name": "drivable"}, 
              {"supercategory": "lane", "id": 2, "name": "lane"}]
count_img = 0
count_ann = 0

for t_json in tqdm(json_files):
    image_name = t_json[:-5] + '.jpg'
    img = cv2.imread(raw_path + image_name)
    image_height, image_width, _ = img.shape

    image = {
        "file_name": image_name,
        "height": image_height,
        "width": image_width,
        "id": count_img
    }

    info = json.load(open(raw_path + t_json))
    shapes = info['shapes']
    imageHeight, imageWidth = info['imageHeight'], info['imageWidth']

    assert (imageHeight == image_height and imageWidth == image_width), 'PLEASE check image shape !!! \n'

    for shape in shapes:
        label = shape['label']
        points = shape['points']
        seg = []

        if label == 'drivable':
            cate_id = 1
        elif label == 'lane':
            cate_id = 2
        else:
            print(f'Error : {label} not fond in category list !!! \n')

        xmin = imageWidth
        ymin = imageHeight
        xmax = 0
        ymax = 0
        for point in points:
            point = [round(point[0], 2), round(point[1], 2)]
            xmin = min(xmin, point[0])
            ymin = min(ymin, point[1])
            xmax = max(xmax, point[0])
            ymax = max(ymax, point[1])
            seg.append(point[0])
            seg.append(point[1])
        assert (xmax > xmin and ymax > ymin), 'BBOX error !!! \n'
        box_w, box_h = round(xmax - xmin, 2), round(ymax - ymin, 2)
        annotation = {
            "segmentation": [seg],
            "area": box_w * box_h,
            "iscrowd": 0,
            "image_id": count_img,
            "bbox": [xmin, ymin, box_w, box_h],
            "category_id": cate_id,
            "id": count_ann
        }
        annotations.append(annotation)
        count_ann += 1
    images.append(image)
    count_img += 1


info = dict()
info['images'] = images
info['annotations'] = annotations
info['categories'] = categories

with open(save_path, 'w') as f:
    json.dump(info, f, indent=4)
print(f'coco format annotation has been writen to : {save_path} \n')

