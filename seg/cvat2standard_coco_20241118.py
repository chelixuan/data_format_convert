import os
import json
from tqdm import tqdm

image_path = "/home/chelx/dataset/seg_images/images/train/train_batch_01_lyon2024/"
train_images = os.listdir(image_path)
print("train images : ", len(train_images))

json_path = "/home/chelx/dataset/seg_images/coco_annotations/raw_coco_anno/"
json_files = [x for x in os.listdir(json_path) if x[-5:] == ".json"]

save_json = "/home/chelx/dataset/seg_images/coco_annotations/train_batch01_lyon2024_202409-202411.json"
# ----------------------------------------------------------------------------
# 标准化标注文档类别对应 -- 跟着标注文档走 
# ----------------------------------------------------------------------------
global standard_cates, ch2eng, ch2cateid
standard_cates = [
        {"id": 1, "name": "step", "supercategory": "step"},
        {"id": 2, "name": "facade", "supercategory": "step"},
        {"id": 3, "name": "floor", "supercategory": "floor"},
        {"id": 4, "name": "slope", "supercategory": "floor"},
        {"id": 5, "name": "wall", "supercategory": "wall"},
        {"id": 6, "name": "bevel", "supercategory": "wall"},
    ]
ch2eng = {
        "台阶平面": "step",
        "台阶立面": "facade",
        "平池底": "floor",
        "斜坡池底": "slope",
        "立面池壁": "wall",
        "斜面池壁": "bevel",
}
ch2cateid = {
    "台阶平面": 1,
    "台阶立面": 2,
    "平池底": 3,
    "斜坡池底": 4,
    "立面池壁": 5,
    "斜面池壁": 6,
}
# ----------------------------------------------------------------------------

# cvat 原始标注 cate_id 与标准化类别对应
def cvatcate_2_standard_id(cvat_cates, cvat_id):
    # [{'id': 1, 'name': '台阶平面', 'supercategory': ''}, {'id': 2, 'name': '台阶立面', 'supercategory': ''}, {'id': 3, 'name': '平池底', 'supercategory': ''}, {'id': 4, 'name': '斜坡池底', 'supercategory': ''}, {'id': 5, 'name': '立面池壁', 'supercategory': ''}, {'id': 6, 'name': '斜面池壁', 'supercategory': ''}]
    """
    cvat_cate: cvat 标注时完整的 categories;
    standard_cates: 标准化标注文档 categories;
    cvat_id: 具体某个 t_ann 在 cvat 标注过程中的 cate_id;
    """
    ch_cate = [x["name"] for x in cvat_cates if x["id"] == cvat_id][0]
    standard_cateid = ch2cateid[ch_cate]
    
    return standard_cateid


collect_images, collect_annos = [], []
count_img, count_anno = 0, 0
for t_json in json_files:
    print(f"\n{t_json} : ")

    f = open(json_path + t_json)
    info = json.load(f)
    images = info["images"]
    cate = info["categories"]
    anno = info["annotations"]

    # image :  {'id': 1, 'width': 1920, 'height': 1080, 
    #           'file_name': 'pool_seg/lyon2024_seg_1028add_bad_case/lyon2024_1028add_badcase_01_0.jpg', 
    #           'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}
    # anno :  {'id': 1, 'image_id': 1, 'category_id': 3, 
    #          'segmentation': [[1.26, 799.47, 470.54, 763.29, 548.39, 758.9, 849.91, 927.76, 1110.3, 1080.0, 0.0, 1080.0, 0.0, 853.43]], 
    #          'area': 256536.0, 'bbox': [0.0, 758.9, 1110.3, 321.1], 'iscrowd': 0, 'attributes': {'occluded': False}}

    imageid_old2new = {}
    for t_image in tqdm(images):
        file_name = t_image["file_name"]
        image_name = file_name[file_name.rfind("/")+1:]
        if image_name in train_images:
            image_id = t_image["id"]
            image_id_new = count_img

            t_image_new = {
                "id": image_id_new,
                "width": t_image["width"],
                "height": t_image["height"],
                "file_name" : "train_batch_01_lyon2024/" + image_name
            }
            imageid_old2new[image_id] = image_id_new
            collect_images.append(t_image_new)
            count_img += 1
    
    for t_ann in tqdm(anno):
        image_id = t_ann["image_id"]
        if image_id in list(imageid_old2new.keys()):
            old_cate_id = t_ann["category_id"]
            new_cate_id = cvatcate_2_standard_id(cate, old_cate_id)
            new_image_id = imageid_old2new[image_id]

            t_ann_new = {
                "id": count_anno, 
                "image_id": new_image_id,
                "category_id": new_cate_id,
                "segmentation": t_ann["segmentation"],
                "area": t_ann["area"],
                "bbox": t_ann["bbox"],
                'iscrowd': 0, 
                'attributes': {'occluded': False},
            }

            count_anno += 1
            collect_annos.append(t_ann_new)


print(f"\nstandard: ")
print("num_image : ", len(collect_images))
print("num_anno : ", len(collect_annos))

info = dict()
info["images"] = collect_images
info["categories"] = standard_cates
info["annotations"] = collect_annos
with open(save_json, "w") as f:
    json.dump(info, f, indent=4)
    print("success ~~ ")
