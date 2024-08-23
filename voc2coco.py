import os
import cv2
import json
from tqdm import tqdm
from collections import Counter
from xml.dom.minidom import parse

class VOC2COCO():
    def __init__(self, root_path, coco_save_path, categories=[]):
        self.root_path = root_path
        self.categories = categories

        files = os.listdir(self.root_path)
        self.images = [x for x in files if '.jpg' in x]
        self.xml_files = [x for x in files if '.xml' in x]

        self.coco_save_path = coco_save_path
        self.coco_images = []
        self.coco_ann = []

        if isinstance(categories, list):
            self.cate_name2id = {t_cate : int(i + 1) for i, t_cate in enumerate(categories)}
        elif isinstance(categories, dict):
            self.cate_name2id = dict((v, k) if isinstance(k, int) else (k, v) for k, v in categories.items())
        self.coco_cate = [{'id': v, 'name': k} for k, v in self.cate_name2id.items()]

        if len(self.categories) > 0:
            self.cate_limit_flag = True
        else:
            self.cate_limit_flag = False


    def readXML(self, path):
        domTree = parse(path)
        rootNode = domTree.documentElement

        image_path = path.strip('xml')+'jpg'
        img = cv2.imread(image_path)
        height = img.shape[0]
        width = img.shape[1]

        objects = rootNode.getElementsByTagName("object")
        bboxes = []
        labels = []

        for object in objects:
            name = object.getElementsByTagName("name")[0]
            name = name.childNodes[0].data
            
            bbox = object.getElementsByTagName("bndbox")[0]
            xmin = bbox.getElementsByTagName("xmin")[0]
            x_min = xmin.childNodes[0].data
            ymin = bbox.getElementsByTagName("ymin")[0]
            y_min = ymin.childNodes[0].data
            xmax = bbox.getElementsByTagName("xmax")[0]
            x_max = xmax.childNodes[0].data
            ymax = bbox.getElementsByTagName("ymax")[0]
            y_max = ymax.childNodes[0].data

            x_min, y_min, x_max, y_max = float(x_min), float(y_min), float(x_max), float(y_max)
            width, height = float(width), float(height)
            
            labels.append(name)
            bbox = [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
        return width, height, labels, bboxes 
    
    def convert_proccess(self,):
        count_image = 0
        count_ann = 0

        all_labels = []

        for xml in tqdm(self.xml_files):
        # for i in tqdm(range(50)):
        #     xml = self.xml_files[i]

            width, height, labels, bboxes = self.readXML(self.root_path + xml)
            image_flag = False

            # 提前给定categories的种类，只使用给定的category
            if self.cate_limit_flag:
                label_used = [x for x in labels if x in self.categories]
                if len(label_used) > 0:
                    image_flag = True
            # 提前未指定使用的categories
            else:
                image_flag = True
            
            if image_flag:
                image_name = xml.strip('xml')+'jpg'
                image_id = count_image
                count_image += 1
                t_image_info = {
                    "file_name": image_name,
                    "height": int(float(height)),
                    "width": int(float(width)),
                    "id": image_id
                }
                self.coco_images.append(t_image_info)

            for i, t_label in enumerate(labels):
                # 只使用提前指定的有限的categories
                if self.cate_limit_flag:
                    # 当前标注bbox的label不在提前指定使用的categories中
                    if t_label not in list(self.cate_name2id.keys()):
                        continue
                    else:
                        t_cate_id = self.cate_name2id[t_label]
                # 提前未指定categories，使用所有有标注的category
                else:
                    if t_label in list(self.cate_name2id.keys()):
                        t_cate_id = self.cate_name2id[t_label]
                    else:
                        used_cate_id = list(self.cate_name2id.values())
                        # category_id 从1开始，默认0代表background
                        if len(used_cate_id) == 0:
                            t_cate_id = 1
                        else:
                            t_cate_id = int(max(used_cate_id) + 1)
                        self.cate_name2id[t_label] = t_cate_id
                        self.coco_cate.append(
                            {
                                'id': t_cate_id,
                                'name': t_label
                            }
                        )

                all_labels.append(t_label)

                x_min, y_min, x_max, y_max = bboxes[i]
                w, h = x_max - x_min, y_max - y_min
                area = w * h

                ann_id = count_ann
                count_ann += 1

                t_ann_info = {
                    "area": area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x_min, y_min, w, h],
                    "category_id": t_cate_id,
                    "id": ann_id
                }
                self.coco_ann.append(t_ann_info)
        
        print()
        print('-' * 80)
        print('num_images = ', len(self.coco_images))
        print('num_ann = ', len(self.coco_ann))
        print()
        print('categories = \n', self.coco_cate)
        print()
        count_cate = Counter(all_labels)
        print('count_cate = ', count_cate)
        print('-' * 80)
        print()
        
        coco = dict()
        coco['images'] = self.coco_images
        coco['annotations'] = self.coco_ann
        coco['categories'] = self.coco_cate

        with open(self.coco_save_path, 'w') as f:
            json.dump(coco, f, indent=4)
            print('coco_format_json file has been writen to : {} \n'.format(self.coco_save_path))
        
        

def main(root_path, coco_save_path, categories):
    voc2coco = VOC2COCO(root_path, coco_save_path, categories)
    voc2coco.convert_proccess()

if __name__ == '__main__':
    raw_path = '/mnt/e/datasets/images/val/0412-0414_batch_0/'
    
    trash_categories = ['person', 'bus', 'car', 'bicycle',]
    trash_coco_json_path = '/mnt/e/dataset/images/annotations/val_0412-0414_batch_0.json'
    main(raw_path, trash_coco_json_path, trash_categories)      

