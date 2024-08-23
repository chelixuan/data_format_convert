import os
import json
import shutil
from tqdm import tqdm
import prettytable as pt


class COCO2YOLO:
    def __init__(self, json_path, txt_path='./yolo_lables/', category_save_path=None, image_path=None):
        """
        json_path : coco format annotation path;
        txt_path: path to save yolo format txts;
        category_save_path: yolo format id2cate table save path, default None;
        """
        self.json_path = json_path
        self.info = json.load(open(self.json_path))
        self.txt_path = txt_path
        os.makedirs(self.txt_path, exist_ok=True)
        self.save_tb_path = category_save_path
        self.image_path = image_path

        self.images = self.info['images']
        self.ann = self.info['annotations']
        self.cates = self.info['categories']

    
    def get_yolo_cate(self):
        cate_id_convert = dict()

        tb = pt.PrettyTable()
        tb.field_names = ['category_id', 'category_name']
        
        for i in range(len(self.cates)):
            cate = self.cates[i]
            ori_cate_id = cate['id']
            new_cate_id = i
            cate_name = cate['name']

            cate_id_convert[ori_cate_id] = new_cate_id

            tb.add_row([new_cate_id, cate_name])

        if self.save_tb_path is not None:
            with open(self.save_tb_path, 'w') as f:
                f.write(str(tb))
                print('yolo category table has been writen to : ', self.save_tb_path, '\n')

        print(tb)
        return cate_id_convert
    
    def get_image_ann(self):
        id2ann = dict()
        for ann in self.ann:
            image_id = ann['image_id']
            bbox = ann['bbox']
            cate_id = ann['category_id']

            temp_key = {
                'coco_bbox': bbox,
                'ori_cate_id': cate_id
            }

            if image_id not in id2ann.keys():
                id2ann[image_id] = [temp_key]
            else:
                id2ann[image_id].append(temp_key)
        
        return id2ann
    
    def get_image_info(self):
        id2info = dict()
        for image in self.images:
            image_id = image['id']
            image_name = image['file_name']
            image_height, image_width = image['height'], image['width']

            id2info[image_id] = {
                'image_name': image_name,
                'image_shape': (image_height, image_width),
            }
        
        return id2info
    
    def write_yolo_txt(self):
        cate_id_convert = self.get_yolo_cate()
        id2info = self.get_image_info()
        id2ann = self.get_image_ann()

        for image in tqdm(self.images):
            image_id = image['id']
            image_name = image['file_name']

            image_info = id2info[image_id]
            image_height, image_width = image_info['image_shape']

            anns = id2ann.get(image_id, None)
            if anns is not None:
                if self.image_path is not None:
                    # shutil.copy(self.image_path + image_name, self.txt_path + image_name)
                    try:
                        # if image exists
                        shutil.copy(self.image_path + image_name, self.txt_path + image_name)
                    except:
                        # print(os.path.exists(self.image_path + image_name))
                        print('image : {} missing ... \n'.format(self.image_path + image_name))
                        continue

                txt_name = image_name[:image_name.rfind('.')] + '.txt'
                with open(self.txt_path + txt_name, 'w') as f:
                    for ann in anns:
                        x, y, w_, h_ = ann['coco_bbox']
                        coco_cate_id = ann['ori_cate_id']

                        yolo_cate_id = cate_id_convert[coco_cate_id]

                        center_x, center_y = (x + 0.5 * w_) / image_width, (y + 0.5 * h_) / image_height
                        w, h = w_ / image_width, h_ / image_height

                        center_x, center_y, w, h = float(round(center_x, 3)), float(round(center_y, 3)), float(round(w, 3)), float(round(h, 3))

                        yolo_info = [yolo_cate_id, center_x, center_y, w, h]
                        line = " ".join(str(x) for x in yolo_info)
                        f.write(line + '\n')


if __name__ == '__main__':
    image_root = '/mnt/e/datasets/dataset/background_parking/20230724_charging/'

    coco_json_path = '/mnt/e/datasets/dataset/background_parking/train_background_parking-coco.json'
    yolo_label_save_path = '/mnt/e/datasets/dataset/background_parking/20230724_charging_yolo/'

    yolo_category_save_path = '/mnt/e/datasets/dataset/background_parking/category.txt'

    coco2yolo = COCO2YOLO(json_path=coco_json_path, txt_path=yolo_label_save_path, category_save_path=yolo_category_save_path, image_path=image_root)
    coco2yolo.write_yolo_txt()

    