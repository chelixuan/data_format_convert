import os
import cv2
import json
import shutil
from tqdm import tqdm

def coco2yolo(image_root, json_path, yolo_txt_path, specific_cates=None):
    info = json.load(open(json_path))
    images = info["images"]
    categories = info["categories"]
    annos = info["annotations"]

    cate_id_convert = dict()
    for t_cate in categories:
        raw_cate_id = t_cate["id"]
        cate_name = t_cate["name"]
        if specific_cates:
            if cate_name in specific_cates:
                new_cate_id = specific_cates.index(cate_name)
                cate_id_convert[raw_cate_id] = new_cate_id
            else:
                print(f"drop category --> id : {raw_cate_id}, name : {cate_name}")
        else: # coco 类别 index 从1开始，yolo 从0开始
            new_cate_id = int(raw_cate_id - 1)
            cate_id_convert[raw_cate_id] = new_cate_id 

    image_id2name = {}
    print(f"\nget image_id2name : ")
    for t_image in tqdm(images):
        file_name = t_image["file_name"]
        image_id = t_image["id"]
        image_id2name[image_id] = file_name

        shutil.copy(image_root + file_name, yolo_txt_path + file_name)

    print(f"\ncoco2yolo : ")
    for t_image_id in tqdm(image_id2name):
        file_name = image_id2name[t_image_id]
        if not os.path.exists(image_root + file_name):
            print(f"image missing : {image_root + file_name}\n")
            continue
        else:
            img = cv2.imread(image_root + file_name)
            image_height, image_width, _ = img.shape

        t_yolo_txt_path = yolo_txt_path + file_name[:file_name.rfind("/") + 1]
        os.makedirs(t_yolo_txt_path, exist_ok=True)
        with open(yolo_txt_path + file_name[:-4] + ".txt", "w") as f:
            for t_ann in annos:
                if t_ann["image_id"] == t_image_id:
                    seg = t_ann["segmentation"][0]
                    raw_cate_id = t_ann["category_id"]
                    new_cate_id = cate_id_convert[raw_cate_id]
                    line = [str(new_cate_id)]
                    for i in range(0, len(seg), 2):
                        x = seg[i]
                        y = seg[i + 1]
                        point_x, point_y = float(round(x / image_width, 3)), float(round(y / image_height, 3))
                        point_x, point_y = min(max(0, point_x), 1), min(max(0, point_y), 1)
                        line.append(str(point_x))
                        line.append(str(point_y))
                    line.append("\n")
                    txt_line = (" ").join(line)
                    f.write(txt_line)
                    

def main():
    image_root = "/home/chelx/dataset/seg_images/images/val/"
    raw_json_path = "/home/chelx/dataset/seg_images/coco_annotations/val_20241118.json"
    yolo_txt_save_path = "/home/chelx/dataset/seg_images/yolo_txt_annos/val/"
    
    os.makedirs(yolo_txt_save_path, exist_ok=True)

    specific_cates = ["step", "facade", "floor", "wall"]

    coco2yolo(image_root, raw_json_path, yolo_txt_save_path, specific_cates)

if __name__ == '__main__':
    main()      

